"""Integration hook: wires a tuned ML transit-speed model (from
scripts/gtfs_realtime/fit_speed_models.py + export_speed_model_onnx.py) into a
CompassApp's generated dataset, so the transit energy models can consume a
per-edge, per-query ML-predicted speed (``transit_speed``) instead of the
default OSM-derived ``edge_speed``.

The ONNX model itself only needs a flat float vector as input (see the
``input_order`` in its manifest JSON). The per-query temporal features
(``hour``/``is_weekday``/``is_peak``) are resolved on the Rust side from
``start_time``/``start_weekday`` query parameters (already sent on every
routing query — see ``deadhead_router.gtfs_time_to_query_time`` and
``GTFSEnergyPredictor._shape_start_times``). Everything else — static
per-edge OSM/GTFS-derived features — must be precomputed here and written as
dense per-edge files (one value per line, in Compass's ``edge_id`` order) for
the ``custom`` traversal model type to load.

Feature engineering here (highway parsing, functional_class mapping, median
imputation, missing-value indicators) mirrors the training-side logic in
scripts/gtfs_realtime/fit_speed_models.py — keep them in sync if either changes.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import tomlkit
from gtfsblocks import Feed

if TYPE_CHECKING:
    from nrel.routee.compass.io.generate_dataset import HookParameters

logger = logging.getLogger("speed_model")

TRANSIT_SPEED_FEATURE_NAME = "transit_speed"


def _parse_maxspeed_mph(val: object) -> float:
    """Parse an OSM maxspeed value to mph. Mirrors realtime_speeds._parse_maxspeed_mph."""
    if val is None:
        return float("nan")
    if isinstance(val, list):
        parsed = [_parse_maxspeed_mph(v) for v in val]
        valid = [v for v in parsed if not np.isnan(v)]
        return min(valid) if valid else float("nan")
    if isinstance(val, (int, float)):
        return float(val) if not np.isnan(float(val)) else float("nan")
    s = str(val).strip().lower()
    if s in ("", "none", "signals", "variable", "walk", "national"):
        return float("nan")
    m = re.match(r"(\d+(?:\.\d+)?)\s*(mph|km/h|kph|kmh)?", s)
    if m:
        speed = float(m.group(1))
        unit = (m.group(2) or "mph").lower().replace("/", "")
        if unit in ("kmh", "kph"):
            speed *= 0.621371
        return speed
    return float("nan")


def _parse_highway(val: object) -> str | None:
    """Return the OSM highway tag as a string. Mirrors realtime_speeds._parse_highway."""
    if val is None:
        return None
    if isinstance(val, list):
        return str(val[0]) if val else None
    return str(val)


def _parse_lanes(val: object) -> float:
    """Parse an OSM lanes value to a float. Mirrors realtime_speeds._parse_lanes."""
    if val is None:
        return float("nan")
    if isinstance(val, list):
        nums: list[int] = []
        for v in val:
            try:
                nums.append(int(str(v)))
            except (ValueError, TypeError):
                pass
        return float(max(nums)) if nums else float("nan")
    try:
        return float(int(str(val)))
    except (ValueError, TypeError):
        return float("nan")


def _compute_n_stops(edges_gdf: gpd.GeoDataFrame, feed: Feed) -> np.ndarray:
    """Count GTFS stops nearest to each edge (spatial join), indexed by edge_id."""
    stops_gdf = gpd.GeoDataFrame(
        feed.stops,
        geometry=gpd.points_from_xy(feed.stops.stop_lon, feed.stops.stop_lat),
        crs="EPSG:4326",
    )
    stops_projected = stops_gdf.to_crs("EPSG:3857")
    edges_projected = edges_gdf[["edge_id", "geometry"]].to_crs("EPSG:3857")
    matched = stops_projected.sjoin_nearest(edges_projected, distance_col="dist")
    counts = matched.groupby("edge_id").size()
    result: np.ndarray = (
        counts.reindex(edges_gdf["edge_id"], fill_value=0).astype(float).to_numpy()
    )
    return result


def compute_transit_speed_features(
    edges_gdf: gpd.GeoDataFrame, feed: Feed, manifest: dict[str, Any]
) -> pd.DataFrame:
    """Build the per-edge feature matrix expected by the tuned speed model.

    Returns a DataFrame indexed by ``edge_id`` (0..N-1, Compass's enumeration
    order) with one column per name in ``manifest["input_order"]`` EXCEPT the
    per-query temporal features (``hour``/``is_weekday``/``is_peak``), which
    are resolved per-query on the Rust side instead.
    """
    cat = manifest["categorical_feature"]
    cat_name = cat["name"]
    numeric_cols = [c for c in manifest["static_per_edge_features"] if c != cat_name]
    one_hot_cols = cat["one_hot_columns"]
    highway_to_functional_class = cat["highway_to_functional_class"]
    functional_class_default = cat["default"]
    feature_medians = manifest.get("feature_medians") or {}
    missing_indicator_features = manifest.get("missing_indicator_features") or []

    edges = edges_gdf.sort_values("edge_id").reset_index(drop=True)

    raw: dict[str, np.ndarray] = {}
    raw["maxspeed_mph"] = edges["maxspeed"].apply(_parse_maxspeed_mph).to_numpy()
    raw["lanes"] = edges["lanes"].apply(_parse_lanes).to_numpy()
    raw["grade"] = pd.to_numeric(edges.get("grade"), errors="coerce").to_numpy()
    raw["grade_abs"] = pd.to_numeric(edges.get("grade_abs"), errors="coerce").to_numpy()
    raw["link_length_km"] = (
        pd.to_numeric(edges["distance"], errors="coerce").to_numpy() / 1000.0
    )
    raw["n_stops"] = _compute_n_stops(edges, feed)
    # No trip-schedule integration for arbitrary network edges (unlike the
    # training archive, which aggregated real observed-trip schedules per
    # road): honestly mark as missing rather than fabricate a value. The
    # model was explicitly trained (via the _was_missing indicators) to
    # handle this.
    raw["scheduled_speed_mph"] = np.full(len(edges), np.nan)

    highway = edges["highway"].apply(_parse_highway)
    functional_class = highway.map(highway_to_functional_class).fillna(
        functional_class_default
    )

    df = pd.DataFrame(raw, index=edges["edge_id"].to_numpy())

    # Missing-value indicators (computed BEFORE imputation).
    for feat in missing_indicator_features:
        df[f"{feat}_was_missing"] = df[feat].isna().astype(float)

    # Median-impute using the exact training-time medians from the manifest.
    for col in numeric_cols:
        if col in df.columns and col in feature_medians:
            df[col] = df[col].fillna(feature_medians[col])

    # One-hot expand functional_class using the manifest's exact column names.
    for col in one_hot_cols:
        label = col[len(cat_name) + 1 :]
        df[col] = (functional_class.to_numpy() == label).astype(float)

    ordered_cols = [c for c in manifest["input_order"] if c in df.columns]
    return df[ordered_cols]


def write_transit_speed_model(
    params: HookParameters,
    feed: Feed,
    onnx_model_path: Path,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    """Hook: write per-edge dense feature files + copy the ONNX model/manifest.

    Returns the list of TOML ``[[search.traversal.models]]`` entries (one
    ``custom`` loader per static feature, plus the ``transit_speed`` model
    itself) that must be inserted BEFORE any ``transit_energy`` block so its
    output is available when the energy model traverses each edge.
    """
    manifest = json.loads(manifest_path.read_text())

    out_onnx = params.output_directory / "transit_speed_model.onnx"
    out_manifest = params.output_directory / "transit_speed_model_manifest.json"
    shutil.copy(onnx_model_path, out_onnx)
    shutil.copy(manifest_path, out_manifest)

    features = compute_transit_speed_features(params.edges, feed, manifest)

    temporal = set(manifest["per_query_temporal_features"])
    model_blocks: list[dict[str, Any]] = []
    for name in features.columns:
        if name in temporal:
            continue
        file_name = f"transit_speed_feature__{name}.txt"
        file_path = params.output_directory / file_name
        np.savetxt(file_path, features[name].to_numpy(), fmt="%.6f")
        model_blocks.append(
            {
                "type": "custom",
                "input_file": file_name,
                # Rust CustomInputFormat has no rename_all, so serde expects
                # the exact PascalCase variant name, not "dense".
                "file_format": "Dense",
                "custom_type": name,
                "variable_config": {"type": "floating_point", "initial": 0.0},
                "accumulator": False,
            }
        )

    model_blocks.append(
        {
            "type": "transit_speed",
            "onnx_model_input_file": out_onnx.name,
            "manifest_input_file": out_manifest.name,
        }
    )
    logger.info(
        f"Wrote {len(model_blocks) - 1} per-edge feature files + transit speed "
        f"ONNX model for {len(features)} edges to {params.output_directory}"
    )
    return model_blocks


def write_routing_config(
    transit_energy_toml_path: Path, routing_toml_path: Path
) -> None:
    """Copy the (not-yet-speed-model-wired) ``transit_energy.toml`` aside for
    use by map matching and deadhead routing.

    Those steps only need cost-comparable candidate paths, not accurate
    predicted speeds — running the ``transit_speed`` ONNX model (and its 19
    per-edge ``custom`` feature loaders) on every edge explored during search
    is pure overhead there, and the ONNX session's internal mutex serializes
    every call across threads (see ``TransitSpeedModelService`` in Rust),
    making it a severe bottleneck under parallel search. Must run BEFORE
    ``insert_transit_speed_config`` mutates ``transit_energy_toml_path``.
    """
    shutil.copy(transit_energy_toml_path, routing_toml_path)
    logger.info(f"Wrote lightweight routing config to {routing_toml_path}")


def insert_transit_speed_config(
    transit_energy_toml_path: Path, model_blocks: list[dict[str, Any]]
) -> None:
    """Insert speed-model TOML blocks before any ``transit_energy`` block, and
    point ``transit_energy`` at the new ``transit_speed`` feature.

    Must run AFTER the ``transit_energy.toml`` file has already been written
    (e.g. by ``gtfs_processing.copy_transit_config``).
    """
    with open(transit_energy_toml_path, "r") as f:
        config = tomlkit.load(f)

    search = config.setdefault("search", tomlkit.table())
    traversal = search.setdefault("traversal", tomlkit.table())
    models = traversal.setdefault("models", tomlkit.aot())

    new_models = tomlkit.aot()
    for block in model_blocks:
        item = tomlkit.item(block)
        new_models.append(item)
    for model in models:
        if model.get("type") == "transit_energy":
            model["speed_feature_name"] = TRANSIT_SPEED_FEATURE_NAME
        new_models.append(model)
    traversal["models"] = new_models

    with open(transit_energy_toml_path, "w") as f:
        tomlkit.dump(config, f)
    logger.info(f"Wired transit speed model into {transit_energy_toml_path}")
