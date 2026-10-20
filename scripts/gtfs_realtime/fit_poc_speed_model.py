"""Train the pooled 12-bin speed POC and compare against a no-time model.

Reads existing archive parquet files only; no downloads or hyperparameter search.
Evaluation holds out whole agencies in three folds. The exported model is then
refit on all agencies. Archive timestamps are naive UTC, converted to local time.
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

from export_speed_model_onnx import (
    SpeedModelManifest,
    build_inference_matrix,
    export_to_onnx,
    validate_onnx,
)
from fit_speed_models import (
    FUNCTIONAL_CLASS_DEFAULT,
    HIGHWAY_TO_FUNCTIONAL_CLASS,
    SPEED_CEIL_MPH,
    SPEED_FLOOR_MPH,
)
from routee.transit.speed_model import (
    TEMPORAL_FEATURES,
    TIME_BIN_FEATURES,
    TIME_BIN_START_HOURS,
    compute_stop_density,
)

log = logging.getLogger(__name__)
NUMERIC_FEATURES = [
    "maxspeed_mph",
    "lanes",
    "grade",
    "grade_abs",
    "link_length_km",
    "n_stops",
]
AGENCY_TIMEZONES = {
    "actransit": "America/Los_Angeles",
    "broward": "America/New_York",
    "cats": "America/New_York",
    "citybus-lafayette": "America/Indiana/Indianapolis",
    "dayton": "America/New_York",
    "gcrta": "America/New_York",
    "metro-transit-madison": "America/Chicago",
    "mountainline": "America/Denver",
    "wrta-youngstown": "America/New_York",
}
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 8,
    "min_samples_leaf": 10,
    "max_features": 0.5,
    "random_state": 42,
    "n_jobs": 4,
}
TRIAL_OVERRIDES = {
    "current": {},
    "all_features": {"max_features": 1.0},
    "deeper": {"max_features": 1.0, "max_depth": 12, "min_samples_leaf": 20},
    "stop_density": {"max_features": 1.0, "max_depth": 12, "min_samples_leaf": 20},
    "current_stop_density": {},
}


def aggregate_agency(df: pd.DataFrame, agency: str, timezone: str) -> pd.DataFrame:
    """Clean observations and aggregate to road, local period and day type."""
    df = df.copy()
    speed = df["mph_moving"]
    df = df[
        np.isfinite(speed)
        & speed.between(SPEED_FLOOR_MPH, SPEED_CEIL_MPH)
        & df["speed_source"].eq("observed")
        & df["n_observations"].gt(0)
    ].copy()
    if df.empty:
        raise ValueError(f"No valid observed speeds for {agency}")
    # Same per-road IQR filter as the archive trainer, without a Python loop.
    quantiles = (
        df.groupby("road_id")["mph_moving"].quantile(np.array([0.25, 0.75])).unstack()
    )
    q1 = df["road_id"].map(quantiles[0.25])
    q3 = df["road_id"].map(quantiles[0.75])
    count = df.groupby("road_id")["mph_moving"].transform("size")
    df = df[
        (count < 4)
        | df["mph_moving"].between(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
    ].copy()
    local = pd.to_datetime(
        df["first_timestamp"], utc=True, errors="raise"
    ).dt.tz_convert(timezone)
    if local.isna().any():
        raise ValueError(f"Missing observation timestamps for {agency}")
    df["time_bin"] = (
        np.searchsorted(TIME_BIN_START_HOURS, local.dt.hour, side="right") - 1
    )
    df["is_weekday"] = (local.dt.dayofweek < 5).astype(int)
    df["weighted_speed"] = df["mph_moving"] * df["n_observations"]
    keys = ["road_id", "time_bin", "is_weekday"]
    grouped = df.groupby(keys)
    agg = grouped.agg(
        weighted_speed=("weighted_speed", "sum"),
        total_observations=("n_observations", "sum"),
        n_trips=("mph_moving", "size"),
    )
    agg["mph_moving_mean"] = agg.pop("weighted_speed") / agg["total_observations"]
    attrs = grouped[NUMERIC_FEATURES + ["highway"]].first()
    agg = agg.join(attrs).reset_index()
    agg = agg[agg["n_trips"] >= 3].copy()
    if agg.empty:
        raise ValueError(f"No bins with at least three observations for {agency}")
    agg["agency"] = agency
    agg["road_id"] = agency + "_" + agg["road_id"].astype(str)
    agg["functional_class"] = (
        agg["highway"].map(HIGHWAY_TO_FUNCTIONAL_CLASS).fillna(FUNCTIONAL_CLASS_DEFAULT)
    )
    for i, name in enumerate(TIME_BIN_FEATURES):
        agg[name] = (agg["time_bin"] == i).astype(float)
    return agg


def make_manifest(
    train: pd.DataFrame, include_stop_density: bool = False
) -> SpeedModelManifest:
    """Describe training-only imputation and the fixed feature vocabulary."""
    categories = [
        f"functional_class_{name}"
        for name in sorted(set(HIGHWAY_TO_FUNCTIONAL_CLASS.values()))
    ]
    static = NUMERIC_FEATURES + (
        ["stop_density_per_km"] if include_stop_density else []
    )
    numeric = static + TEMPORAL_FEATURES
    medians = train[numeric].replace([np.inf, -np.inf], np.nan).median()
    if medians.isna().any():
        raise ValueError(
            f"Entirely missing training features: {list(medians[medians.isna()].index)}"
        )
    return {
        "target": "mph_moving_mean (observation-weighted mean moving speed, mph)",
        "input_order": numeric
        + categories
        + [f"{name}_was_missing" for name in static],
        "static_per_edge_features": static + ["functional_class"],
        "per_query_temporal_features": TEMPORAL_FEATURES,
        "time_bin_start_hours": list(TIME_BIN_START_HOURS),
        "trace_rounding_mph": 0.1,
        "categorical_feature": {
            "name": "functional_class",
            "one_hot_columns": categories,
            "highway_to_functional_class": HIGHWAY_TO_FUNCTIONAL_CLASS,
            "default": FUNCTIONAL_CLASS_DEFAULT,
        },
        "feature_medians": {name: float(value) for name, value in medians.items()},
        "missing_indicator_features": static,
    }


def balanced_weights(df: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Give agencies equal total weight while retaining observation weights."""
    weights = df["total_observations"].astype(float)
    totals = weights.groupby(df["agency"]).transform("sum")
    return (weights / totals * len(df) / df["agency"].nunique()).to_numpy()


def load_training_data(root: Path) -> pd.DataFrame:
    """Load and aggregate archive observations with local-time bins."""
    paths = sorted(root.glob("agency=*"))
    if len(paths) < 3:
        raise ValueError("POC training requires at least three agencies")
    unknown = [
        p.name.split("=", 1)[1]
        for p in paths
        if p.name.split("=", 1)[1] not in AGENCY_TIMEZONES
    ]
    if unknown:
        raise ValueError(f"Add verified agency timezones before training: {unknown}")
    agencies = []
    columns = NUMERIC_FEATURES + [
        "highway",
        "road_id",
        "mph_moving",
        "n_observations",
        "speed_source",
        "first_timestamp",
    ]
    for path in paths:
        agency = path.name.split("=", 1)[1]
        log.info("Loading and aggregating %s", agency)
        frame = pd.read_parquet(path, columns=columns)
        aggregated = aggregate_agency(frame, agency, AGENCY_TIMEZONES[agency])
        log.info(
            "%s: %d observations -> %d road/period/day-type bins",
            agency,
            len(frame),
            len(aggregated),
        )
        agencies.append(aggregated)
    data = pd.concat(agencies, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    return data


def train(
    root: Path,
    output: Path,
    trial: str = "current",
    aggregated_csv: Path | None = None,
) -> None:
    """Compare time/no-time models for one fixed trial, then export the time model."""
    params = {**MODEL_PARAMS, **TRIAL_OVERRIDES[trial]}
    include_stop_density = trial in ("stop_density", "current_stop_density")
    data = pd.read_csv(aggregated_csv) if aggregated_csv else load_training_data(root)
    if include_stop_density:
        data["stop_density_per_km"] = compute_stop_density(
            data["n_stops"], data["link_length_km"]
        )
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "aggregated_training_data.csv", index=False)
    log.info("Trial %s: %s; stop density=%s", trial, params, include_stop_density)
    log.info("Combined %d bins from %d agencies", len(data), data["agency"].nunique())
    metrics: list[dict[str, str | float]] = []
    for fold, (train_idx, test_idx) in enumerate(
        GroupKFold(3).split(data, groups=data["agency"]), 1
    ):
        training, test = data.iloc[train_idx], data.iloc[test_idx]
        manifest = make_manifest(training, include_stop_density)
        x_train = build_inference_matrix(training, manifest)
        x_test = build_inference_matrix(test, manifest)
        log.info(
            "Fold %d, held-out agencies: %s", fold, sorted(test["agency"].unique())
        )
        for label, use_time in [("12_bin", True), ("no_time", False)]:
            keep = [
                i
                for i, name in enumerate(manifest["input_order"])
                if use_time or name not in TEMPORAL_FEATURES
            ]
            model = RandomForestRegressor(**params)
            model.fit(
                x_train[:, keep],
                training["mph_moving_mean"],
                sample_weight=balanced_weights(training),
            )
            predictions = model.predict(x_test[:, keep])
            for agency in sorted(test["agency"].unique()):
                mask = test["agency"].eq(agency).to_numpy()
                actual = test.loc[mask, "mph_moving_mean"]
                weights = test.loc[mask, "total_observations"]
                metrics.append(
                    {
                        "model": label,
                        "agency": agency,
                        "mae_mph": float(
                            mean_absolute_error(
                                actual, predictions[mask], sample_weight=weights
                            )
                        ),
                        "rmse_mph": float(
                            np.sqrt(
                                mean_squared_error(
                                    actual, predictions[mask], sample_weight=weights
                                )
                            )
                        ),
                    }
                )
    comparison = pd.DataFrame(metrics)
    comparison.to_csv(output / "time_comparison.csv", index=False)
    summary = comparison.groupby("model")[["mae_mph", "rmse_mph"]].mean()
    log.info("Equal-agency averages:\n%s", summary.to_string())
    manifest = make_manifest(data, include_stop_density)
    manifest["training"] = {
        "agencies": sorted(data["agency"].unique()),
        "agency_timezones": AGENCY_TIMEZONES,
        "model_params": params,
        "trial": trial,
        "stop_density_min_length_km": 0.05 if include_stop_density else None,
        "weighting": "equal total weight per agency, observation-weighted within agency",
        "comparison": summary.reset_index().to_dict(orient="records"),
    }
    log.info("Refitting 12-bin model on all agencies")
    model = RandomForestRegressor(**params)
    model.fit(
        build_inference_matrix(data, manifest),
        data["mph_moving_mean"],
        sample_weight=balanced_weights(data),
    )
    model_path = output / "tuned_rf_speed_model.joblib"
    manifest_path = output / "tuned_rf_speed_model_manifest.json"
    onnx_path = output / "tuned_rf_speed_model.onnx"
    joblib.dump({"model": model}, model_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    export_to_onnx(model_path, manifest_path, onnx_path)
    validate_onnx(
        onnx_path, model_path, manifest_path, output / "aggregated_training_data.csv"
    )
    log.info("Model ready in %s", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--link-observations-root",
        type=Path,
        default=Path("gtfsrt_archive_runs/link_observations"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/speed_poc"))
    parser.add_argument("--trial", choices=list(TRIAL_OVERRIDES), default="current")
    parser.add_argument(
        "--aggregated-csv",
        type=Path,
        help="Reuse POC aggregated training rows instead of reading the archive",
    )
    args = parser.parse_args()
    train(args.link_observations_root, args.output_dir, args.trial, args.aggregated_csv)
