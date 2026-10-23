"""Precompute six daily periods for weekday/weekend speeds on road links.

ONNX inference runs only during dataset generation, in bounded batches. Speeds
are rounded to 0.1 mph and deduplicated into a dense edge-to-trace map.
Rust selects speeds using departure time plus elapsed search time; no inference
session or static feature loaders are retained in the routing application.

Feature engineering matches scripts/gtfs_realtime/fit_poc_speed_model.py.
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
import numpy.typing as npt
import onnxruntime as ort
import pandas as pd
import tomlkit
from gtfsblocks import Feed

if TYPE_CHECKING:
    from nrel.routee.compass.io.generate_dataset import HookParameters

logger = logging.getLogger("speed_model")

TRANSIT_SPEED_FEATURE_NAME = "transit_speed"
TIME_BIN_START_HOURS = (0, 6, 9, 15, 19, 22)
TIME_BIN_FEATURES = [f"time_bin_{i}" for i in range(len(TIME_BIN_START_HOURS))]
TEMPORAL_FEATURES = TIME_BIN_FEATURES + ["is_weekday"]
SPEED_TRACE_MODEL_CONFIG: dict[str, Any] = {
    "type": "transit_speed",
    "format_version": 2,
    "edge_trace_input_file": "transit_speed_edge_traces.bin",
    "traces_input_file": "transit_speed_traces.bin",
}
DEFAULT_SPEED_MODEL_DIR = (
    Path(__file__).parent / "resources" / "speed_models" / "random-forest"
)


def compute_stop_density(n_stops: pd.Series, link_length_km: pd.Series) -> pd.Series:
    """Return stops/km, treating links shorter than 50 m as 50 m.

    Missing or nonpositive lengths produce missing values for model imputation.
    """
    length = link_length_km.where(link_length_km > 0).clip(lower=0.05)
    return n_stops / length


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


def _compute_n_stops(
    edges_gdf: gpd.GeoDataFrame, feed: Feed
) -> npt.NDArray[np.float64]:
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
    result: npt.NDArray[np.float64] = (
        counts.reindex(edges_gdf["edge_id"], fill_value=0).astype(float).to_numpy()
    )
    return result


def compute_transit_speed_features(
    edges_gdf: gpd.GeoDataFrame, feed: Feed, manifest: dict[str, Any]
) -> pd.DataFrame:
    """Build the per-edge feature matrix expected by the tuned speed model.

    Returns a DataFrame indexed by ``edge_id`` (0..N-1, Compass's enumeration
    order) with one column per name in ``manifest["input_order"]`` EXCEPT the
    temporal features (period indicators and ``is_weekday``), which are added
    for each discrete time bin during preprocessing.
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

    raw: dict[str, npt.NDArray[np.float64]] = {}
    raw["maxspeed_mph"] = edges["maxspeed"].apply(_parse_maxspeed_mph).to_numpy()
    raw["lanes"] = edges["lanes"].apply(_parse_lanes).to_numpy()
    raw["grade"] = pd.to_numeric(edges.get("grade"), errors="coerce").to_numpy()
    raw["grade_abs"] = pd.to_numeric(edges.get("grade_abs"), errors="coerce").to_numpy()
    raw["link_length_km"] = (
        pd.to_numeric(edges["distance"], errors="coerce").to_numpy() / 1000.0
    )
    raw["n_stops"] = _compute_n_stops(edges, feed)
    if "stop_density_per_km" in numeric_cols:
        raw["stop_density_per_km"] = compute_stop_density(
            pd.Series(raw["n_stops"]), pd.Series(raw["link_length_km"])
        ).to_numpy()

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


def patch_vehicle_speed_feature(
    vehicles_dir: Path,
    vehicle_names: list[str],
    speed_feature_name: str = TRANSIT_SPEED_FEATURE_NAME,
) -> None:
    """Repoint each named vehicle's own energy-rate model at ``transit_speed``.

    ``vehicles_dir`` also holds every OTHER bundled vehicle model shipped by
    ``nrel.routee.compass`` (Mazda 3, Ford Focus, Tesla, etc.) copied in by
    ``generate_compass_dataset``'s "copying vehicle configuration files" step
    — only the transit bus model(s) actually selected via ``vehicle_models``
    should be touched, so ``vehicle_names`` scopes the patch to those.

    The ``speed_feature_name`` override on the ``transit_energy`` traversal
    model only affects the wrapper's OWN stop-penalty kinetic-energy calc
    (see ``TransitIceEnergyModel``/``TransitBevEnergyModel`` in Rust) — the
    vehicle's underlying per-mile energy-rate model (smartcore/ONNX,
    interpolated by speed + grade) reads its OWN ``input_features`` list,
    which hardcodes ``"name": "edge_speed"`` in the bundled vehicle JSON
    templates (from the ``nrel.routee.compass`` package). Without this patch,
    the ML transit speed only affects stop penalties, not the primary energy
    consumption rate. Must run AFTER vehicle JSON files are copied into
    ``vehicles_dir`` (e.g. by ``generate_compass_dataset``'s "copying vehicle
    configuration files" step, which runs before hooks).
    """
    for vehicle_path in sorted(vehicles_dir.glob("*.json")):
        if vehicle_path.stem not in vehicle_names:
            continue
        vehicle = json.loads(vehicle_path.read_text())
        changed = False
        for feature in vehicle.get("input_features", []):
            if feature.get("type") == "speed" and feature.get("name") == "edge_speed":
                feature["name"] = speed_feature_name
                changed = True
        feature_bounds = (
            vehicle.get("model_type", {})
            .get("interpolate", {})
            .get("feature_bounds", {})
        )
        if "edge_speed" in feature_bounds:
            feature_bounds[speed_feature_name] = feature_bounds.pop("edge_speed")
            changed = True
        if changed:
            vehicle_path.write_text(json.dumps(vehicle, indent=2))
            logger.info(f"Repointed {vehicle_path.name} at '{speed_feature_name}'")


def write_transit_speed_model(
    params: HookParameters,
    feed: Feed,
    onnx_model_path: Path,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    """Precompute deduplicated traces and return the speed traversal config."""
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("time_bin_start_hours") != list(TIME_BIN_START_HOURS):
        raise ValueError(
            "Speed model must use the six-period POC time bins; retrain with "
            "scripts/gtfs_realtime/fit_poc_speed_model.py"
        )

    out_onnx = params.output_directory / "transit_speed_model.onnx"
    out_manifest = params.output_directory / "transit_speed_model_manifest.json"
    shutil.copy(onnx_model_path, out_onnx)
    shutil.copy(manifest_path, out_manifest)

    features = compute_transit_speed_features(params.edges, feed, manifest)
    logger.info("Precomputing 12 speed bins for %d road links", len(features))

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    _write_speed_traces(
        features, manifest["input_order"], session, params.output_directory
    )
    return [SPEED_TRACE_MODEL_CONFIG.copy()]


def _write_speed_traces(
    features: pd.DataFrame,
    input_order: list[str],
    session: ort.InferenceSession,
    output_directory: Path,
    batch_size: int = 4096,
) -> None:
    """Write uint32 IDs and 12-float32 mph traces, rounded to 0.1 mph."""
    if batch_size <= 0 or features.empty:
        raise ValueError("Speed preprocessing requires edges and a positive batch size")
    if not np.array_equal(features.index.to_numpy(), np.arange(len(features))):
        raise ValueError("Speed features must be in dense Compass edge_id order")
    temporal = set(TEMPORAL_FEATURES)
    if not temporal.issubset(input_order) or len(set(input_order)) != len(input_order):
        raise ValueError(
            "Speed manifest must contain unique inputs including time features"
        )
    static_columns = [name for name in input_order if name not in temporal]
    static_indices = [input_order.index(name) for name in static_columns]
    trace_ids: dict[bytes, int] = {}
    with (
        (output_directory / "transit_speed_edge_traces.bin").open("wb") as mapping,
        (output_directory / "transit_speed_traces.bin").open("wb") as traces_file,
    ):
        for start in range(0, len(features), batch_size):
            batch = features.iloc[start : start + batch_size]
            inputs = np.empty((len(batch), len(input_order)), dtype=np.float32)
            inputs[:, static_indices] = batch[static_columns].to_numpy(dtype=np.float32)
            traces = np.empty((len(batch), 12), dtype="<f4")
            for day_type in range(2):
                inputs[:, input_order.index("is_weekday")] = float(day_type == 0)
                for period in range(6):
                    for i, feature in enumerate(TIME_BIN_FEATURES):
                        inputs[:, input_order.index(feature)] = float(i == period)
                    predictions = np.asarray(
                        session.run(None, {"input": inputs})[0], dtype=np.float32
                    )
                    if predictions.shape not in ((len(batch),), (len(batch), 1)):
                        raise ValueError(
                            f"Speed model returned invalid output shape {predictions.shape}"
                        )
                    if not np.isfinite(predictions).all():
                        raise ValueError("Speed model returned non-finite predictions")
                    traces[:, day_type * 6 + period] = np.round(
                        np.maximum(predictions.reshape(-1), 1.0), 1
                    )
            ids = np.empty(len(batch), dtype="<u4")
            for i, trace in enumerate(traces):
                key = trace.tobytes()
                trace_id = trace_ids.get(key)
                if trace_id is None:
                    trace_id = len(trace_ids)
                    trace_ids[key] = trace_id
                    traces_file.write(key)
                ids[i] = trace_id
            mapping.write(ids.tobytes())
    logger.info(
        "Precomputed %d speed bins for %d edges (%d unique traces)",
        12,
        len(features),
        len(trace_ids),
    )


def insert_transit_speed_config(
    transit_energy_toml_path: Path, model_blocks: list[dict[str, Any]]
) -> None:
    """Replace posted speeds before time accumulation and energy evaluation.

    Both ``edge_speed`` and its ``transit_speed`` alias come from the same
    lookup, so routing costs, elapsed time and energy use consistent speeds.
    """
    with open(transit_energy_toml_path, "r") as f:
        config = tomlkit.load(f)

    search = config.setdefault("search", tomlkit.table())
    traversal = search.setdefault("traversal", tomlkit.table())
    models = traversal.setdefault("models", tomlkit.aot())

    new_models = tomlkit.aot()
    speed_models = [model for model in models if model.get("type") == "speed"]
    if len(speed_models) != 1:
        raise ValueError("Transit config must have exactly one posted-speed model")
    for model in models:
        if model.get("type") == "speed":
            for block in model_blocks:
                new_models.append(tomlkit.item(block))
            continue
        if model.get("type") == "transit_energy":
            model["speed_feature_name"] = TRANSIT_SPEED_FEATURE_NAME
        new_models.append(model)
    traversal["models"] = new_models

    with open(transit_energy_toml_path, "w") as f:
        tomlkit.dump(config, f)
    logger.info(f"Wired transit speed model into {transit_energy_toml_path}")
