"""Fit speed prediction models to aggregated realtime link speed data.

Reads combined per-trip link speed CSVs from one or more transit agency
directories, removes outliers, aggregates to mean speed per road link x
time-of-day bin x weekday/weekend, and trains regression models using only
features available from OSM (generalizable to any US transit agency).

0. Baseline  — speed_limit x constant
1. Linear Regression (OLS)I 
2. Random Forest
3. Histogram Gradient Boosting (handles NaN natively)

The train/test split is spatial — entire road segments are held out — so
that metrics reflect the real use-case of predicting speeds on *unseen*
roads in *both* networks.

Usage
-----
    # Single agency
    python fit_speed_models.py --data-dir reports/realtime/greater_portland_me

    # Multiple agencies (combined training)
    python fit_speed_models.py \\
        --data-dir reports/realtime/greater_portland_me \\
        --data-dir reports/realtime/cdta_albany
"""

import argparse
import json
import logging
import textwrap
from pathlib import Path
from typing import Any, TypedDict, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_DATA_DIRS = [
    Path("reports/realtime/greater_portland_me"),
    Path("reports/realtime/cdta_albany"),
]

# Only features derivable from OSM — no agency- or trip-specific data.
NUMERIC_FEATURES = [
    "maxspeed_mph",
    "lanes",
    "grade",
    "grade_abs",
    "link_length_km",
    "n_stops",
    "scheduled_speed_mph",
]
CATEGORICAL_FEATURES = ["highway"]
TEMPORAL_FEATURES = ["hour", "is_weekday", "is_peak"]
TARGET = "mph_moving"

# Outlier thresholds
SPEED_FLOOR_MPH = 1.0  # below this is likely GPS noise / dwell misattribution
SPEED_CEIL_MPH = 65.0  # above this is implausible for transit buses
IQR_MULTIPLIER = 1.5  # per-road IQR fence

# FHWA-style functional-class grouping for the OSM `highway` tag: coarsens the
# ~15 raw subtypes into a small, closed vocabulary that's more consistent across
# agencies/regions with differing OSM tagging conventions (used by the tuned
# HGB model in fit_and_evaluate_models).
HIGHWAY_TO_FUNCTIONAL_CLASS: dict[str, str] = {
    "motorway": "freeway",
    "motorway_link": "freeway",
    "trunk": "freeway",
    "trunk_link": "freeway",
    "primary": "principal_arterial",
    "primary_link": "principal_arterial",
    "secondary": "minor_arterial",
    "secondary_link": "minor_arterial",
    "tertiary": "collector",
    "tertiary_link": "collector",
    "unclassified": "collector",
    "residential": "local",
    "living_street": "local",
    "busway": "local",
    "service": "local",
}
FUNCTIONAL_CLASS_DEFAULT = "local"

# A tuned estimator's hyperparameter values (e.g. max_depth=None, learning_rate=0.05).
HyperParamValue = int | float | str | None
# sklearn ships no type stubs, so its estimator classes/instances are unavoidably Any.


class ModelMetrics(TypedDict):
    """Regression metrics returned by :func:`evaluate_model`."""

    model: str
    r2: float
    rmse_mph: float
    mae_mph: float


class TunedModelConfig(TypedDict):
    """One entry of :data:`TUNED_MODEL_CONFIGS`."""

    label: str
    estimator_cls: Any
    needs_imputation: bool
    param_distributions: dict[str, list[HyperParamValue]]


def _resolve_input(data_dir: Path) -> Path:
    """Find the best available per-trip speed CSV in a single agency directory."""
    all_days = data_dir / "realtime_link_speeds_all_days.csv"
    if all_days.exists():
        return all_days
    candidates = sorted(data_dir.glob("realtime_link_speeds_2*.csv"))
    candidates = [f for f in candidates if "aggregated" not in f.name]
    if not candidates:
        raise FileNotFoundError(f"No per-trip speed CSVs found in {data_dir}")
    return candidates[-1]


def load_and_clean(csv_path: Path, agency_label: str | None = None) -> pd.DataFrame:
    """Load per-trip link speeds and apply basic sanity filters."""
    log.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)
    log.info("  Raw rows: %d", len(df))

    df = df.dropna(subset=[TARGET])
    df = df[np.isfinite(df[TARGET].to_numpy(dtype=float))]
    # Keep only directly observed links (≥2 GPS pings on the link)
    if "speed_source" in df.columns:
        df = df[df["speed_source"] == "observed"]
    # Hard floor/ceiling
    df = df[(df[TARGET] >= SPEED_FLOOR_MPH) & (df[TARGET] <= SPEED_CEIL_MPH)]
    log.info("  After basic filters: %d rows", len(df))

    # Tag agency so road_ids from different networks don't collide
    if agency_label:
        df["agency"] = agency_label
        df["road_id"] = agency_label + "_" + df["road_id"].astype(str)

    return cast(pd.DataFrame, df)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove per-road outliers using the IQR method.

    For each road_id, speeds outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR] are
    removed. Roads with fewer than 4 observations skip IQR filtering (the
    hard floor/ceiling is still in effect).
    """
    n_before = len(df)
    keep_mask: pd.Series = pd.Series(True, index=df.index, dtype=bool)

    for road_id, group in df.groupby("road_id"):
        if len(group) < 4:
            continue
        q1 = group[TARGET].quantile(0.25)
        q3 = group[TARGET].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - IQR_MULTIPLIER * iqr
        hi = q3 + IQR_MULTIPLIER * iqr
        keep_mask.loc[group.index] = group[TARGET].between(lo, hi)

    df = df[keep_mask]
    n_removed = n_before - len(df)
    log.info(
        "Outlier removal: %d rows removed (%.1f%%), %d remaining",
        n_removed,
        100 * n_removed / max(n_before, 1),
        len(df),
    )
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hour, is_weekday, and is_peak from first_timestamp."""
    if "first_timestamp" not in df.columns:
        df["hour"] = np.nan
        df["is_weekday"] = 1
        df["is_peak"] = 0
        return df

    ts = pd.to_datetime(df["first_timestamp"], errors="coerce")
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek  # 0=Mon … 6=Sun
    df["is_weekday"] = (df["day_of_week"] < 5).astype(int)
    df["is_peak"] = df["hour"].apply(
        lambda h: 1 if (7 <= h <= 9 or 16 <= h <= 18) else 0
    )
    return df


def aggregate_to_road_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-trip observations to mean speed per (road, hour, is_weekday).

    This is the modeling unit: "what average speed should we expect on this
    road segment during this hour on a weekday/weekend?"

    Weighted by n_observations (links with more GPS pings contribute more).
    """
    group_cols = ["road_id", "hour", "is_weekday"]
    if "agency" in df.columns:
        group_cols = ["agency"] + group_cols
    # Road-level attributes (constant per road_id) — carried forward via first()
    road_attrs = [
        "highway",
        "maxspeed_mph",
        "lanes",
        "grade",
        "grade_abs",
        "link_length_km",
        "n_stops",
    ]
    road_attrs = [c for c in road_attrs if c in df.columns]

    # scheduled_speed_mph varies by time-of-day — aggregate as weighted mean
    has_sched_speed = "scheduled_speed_mph" in df.columns

    def _weighted_mean(g: pd.DataFrame) -> pd.Series:
        w = g["n_observations"].to_numpy(dtype=float)
        total_w = w.sum()
        if total_w == 0:
            w = np.ones(len(g))
            total_w = float(len(g))
        result = {
            "mph_moving_mean": np.average(g[TARGET].to_numpy(dtype=float), weights=w),
            "mph_moving_std": g[TARGET].std(),
            "n_trips": len(g),
            "total_observations": int(total_w),
        }
        if has_sched_speed:
            valid = g["scheduled_speed_mph"].notna()
            if valid.any():
                result["scheduled_speed_mph"] = np.average(
                    g.loc[valid, "scheduled_speed_mph"].to_numpy(dtype=float),
                    weights=w[valid.to_numpy(dtype=bool)],
                )
            else:
                result["scheduled_speed_mph"] = np.nan
        return pd.Series(result)

    agg = (
        df.groupby(group_cols)
        .apply(_weighted_mean, include_groups=False)  # type: ignore[call-overload]
        .reset_index()
    )

    # Attach road-level attributes from first occurrence
    road_props = df.groupby("road_id")[road_attrs].first().reset_index()
    agg = agg.merge(road_props, on="road_id", how="left")

    # is_peak is derivable from hour
    agg["is_peak"] = agg["hour"].apply(
        lambda h: 1 if (7 <= h <= 9 or 16 <= h <= 18) else 0
    )

    log.info(
        "Aggregated to %d (road x hour x weekday/weekend) groups from %d trips",
        len(agg),
        df["trip_id"].nunique() if "trip_id" in df.columns else -1,
    )
    return cast(pd.DataFrame, agg)


def build_feature_matrix(
    df: pd.DataFrame,
    encoder: OneHotEncoder | None = None,
    fit: bool = False,
    categorical_features: list[str] | None = None,
) -> tuple[np.ndarray, OneHotEncoder]:
    """Build the feature matrix X from an aggregated DataFrame.

    *categorical_features* defaults to ``CATEGORICAL_FEATURES``; pass a
    different list (e.g. ``["functional_class"]``) to swap in an alternative
    categorical encoding without touching the numeric/temporal features.
    """
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    num_cols = NUMERIC_FEATURES + TEMPORAL_FEATURES
    X_num = df[num_cols].values.astype(float)

    cat_data = df[categorical_features].fillna("unknown").values
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    if fit:
        X_cat = encoder.fit_transform(cat_data)
    else:
        X_cat = encoder.transform(cat_data)

    X = np.hstack([X_num, X_cat])
    return X, encoder


def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None = None,
) -> ModelMetrics:
    """Compute regression metrics (optionally observation-weighted)."""
    r2 = r2_score(y_true, y_pred, sample_weight=weights)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=weights))
    mae = mean_absolute_error(y_true, y_pred, sample_weight=weights)
    log.info("  %-35s  R²=%.4f  RMSE=%.2f mph  MAE=%.2f mph", name, r2, rmse, mae)
    return {"model": name, "r2": r2, "rmse_mph": rmse, "mae_mph": mae}


# Hyperparameter search spaces for each tuned, functional_class-enabled model.
# HistGradientBoostingRegressor currently cannot be exported to ONNX with this
# environment's skl2onnx/sklearn versions (a boolean-attribute serialization bug
# in its missing-value tree nodes); RandomForestRegressor and
# GradientBoostingRegressor are kept as ONNX-exportable alternatives.
TUNED_MODEL_CONFIGS: dict[str, TunedModelConfig] = {
    "hgb": {
        "label": "Histogram Gradient Boosting (tuned, functional_class)",
        "estimator_cls": HistGradientBoostingRegressor,
        # HGB handles NaN natively; feed it the raw (unimputed) matrix.
        "needs_imputation": False,
        "param_distributions": {
            "max_iter": [100, 200, 300, 500, 800],
            "max_depth": [3, 4, 5, 6, 8, None],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "min_samples_leaf": [5, 10, 20, 30, 50],
            "l2_regularization": [0.0, 0.1, 0.5, 1.0],
            "max_leaf_nodes": [15, 31, 63, 127, None],
        },
    },
    "rf": {
        "label": "Random Forest (tuned, functional_class)",
        "estimator_cls": RandomForestRegressor,
        # Plain RandomForestRegressor can't handle NaN; needs imputed input.
        "needs_imputation": True,
        "param_distributions": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [8, 12, 16, 20, None],
            "min_samples_leaf": [1, 2, 5, 10, 20],
            "max_features": ["sqrt", "log2", 0.5, 1.0],
        },
    },
    "gbr": {
        "label": "Gradient Boosting (tuned, functional_class)",
        "estimator_cls": GradientBoostingRegressor,
        # Classic GradientBoostingRegressor can't handle NaN; needs imputed input.
        "needs_imputation": True,
        "param_distributions": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [2, 3, 4, 6],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "min_samples_leaf": [5, 10, 20, 30],
            "subsample": [0.6, 0.8, 1.0],
        },
    },
}


def _tune_and_persist_model(
    model_key: str,
    config: TunedModelConfig,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray,
    w_test: np.ndarray,
    groups_train: np.ndarray,
    tune_n_iter: int,
    tune_cv_splits: int,
    tuned_encoder: OneHotEncoder,
    tuned_cat_names: list[str],
    output_dir: Path,
    results: list[ModelMetrics],
    feature_medians: dict[str, float] | None = None,
    missing_indicator_features: list[str] | None = None,
    fixed_params: dict[str, HyperParamValue] | None = None,
) -> tuple[Any, np.ndarray, dict[str, HyperParamValue]]:
    """Spatial-CV hyperparameter search + weighted refit for one estimator.

    Evaluates the tuned model on the held-out test set (appending to *results*)
    and persists it (model + feature manifest) for downstream reuse (e.g. ONNX
    export). *feature_medians* (input_order -> train-median value), if given,
    is recorded in the manifest so inference-time code reproduces the same
    missing-value imputation the model was trained on. *missing_indicator_features*
    (subset of NUMERIC_FEATURES), if given, records that *X_train*/*X_test*
    carry extra trailing ``{feature}_was_missing`` columns (1.0 where the raw
    value was NaN before imputation), so the model can still distinguish
    originally-missing values instead of treating the imputed median as real.
    If *fixed_params* is given, the ``RandomizedSearchCV`` step is skipped
    entirely and the estimator is refit directly with those hyperparameters
    (e.g. to reuse a previous search's best params on a new feature set).
    Returns ``(fitted_model, y_pred_test, best_params)``.
    """
    estimator_cls = config["estimator_cls"]
    label = config["label"]

    if fixed_params is not None:
        log.info(
            "Fitting %s with fixed params (skipping hyperparameter search): %s",
            estimator_cls.__name__,
            fixed_params,
        )
        best_params = fixed_params
    else:
        log.info(
            "Fitting tuned %s (functional_class feature + spatial GroupKFold "
            "search, n_iter=%d, cv=%d folds) …",
            estimator_cls.__name__,
            tune_n_iter,
            tune_cv_splits,
        )
        # Search unweighted (avoids sample_weight/metadata-routing pitfalls in
        # CV scoring), then refit the winning params with observation weights
        # to stay consistent with the other models.
        search = RandomizedSearchCV(
            estimator=estimator_cls(random_state=42),
            param_distributions=config["param_distributions"],
            n_iter=tune_n_iter,
            scoring="r2",
            cv=GroupKFold(n_splits=tune_cv_splits),
            n_jobs=-1,
            random_state=42,
            refit=False,
        )
        search.fit(X_train, y_train, groups=groups_train)
        log.info("  Best params: %s", search.best_params_)
        log.info("  Best CV R² (unweighted): %.4f", search.best_score_)
        best_params = search.best_params_

    model = estimator_cls(random_state=42, **best_params)
    model.fit(X_train, y_train, sample_weight=w_train)
    y_pred = model.predict(X_test)
    results.append(evaluate_model(label, y_test, y_pred, w_test))

    # Persist the fitted model + a feature manifest so it can be used for
    # inference (e.g. ONNX export for the transit energy pipeline) without
    # retraining. Input order matches build_feature_matrix's
    # np.hstack([X_num, X_cat]): NUMERIC_FEATURES + TEMPORAL_FEATURES + one-hot
    # functional_class columns, plus trailing missing-indicator columns if any.
    model_path = output_dir / f"tuned_{model_key}_speed_model.joblib"
    joblib.dump(
        {
            "model": model,
            "encoder": tuned_encoder,
            "numeric_features": NUMERIC_FEATURES,
            "temporal_features": TEMPORAL_FEATURES,
            "categorical_feature": "functional_class",
            "category_names": tuned_cat_names,
            "highway_to_functional_class": HIGHWAY_TO_FUNCTIONAL_CLASS,
            "functional_class_default": FUNCTIONAL_CLASS_DEFAULT,
        },
        model_path,
    )
    indicator_cols = [f"{c}_was_missing" for c in (missing_indicator_features or [])]
    manifest = {
        "target": "mph_moving_mean (observation-weighted mean moving speed, mph)",
        "input_order": NUMERIC_FEATURES
        + TEMPORAL_FEATURES
        + tuned_cat_names
        + indicator_cols,
        "static_per_edge_features": NUMERIC_FEATURES + ["functional_class"],
        "per_query_temporal_features": TEMPORAL_FEATURES,
        "categorical_feature": {
            "name": "functional_class",
            "one_hot_columns": tuned_cat_names,
            "highway_to_functional_class": HIGHWAY_TO_FUNCTIONAL_CLASS,
            "default": FUNCTIONAL_CLASS_DEFAULT,
        },
        # If set, inference-time code MUST fill missing/NaN feature values with
        # these train-set medians before predicting — this model was trained on
        # median-imputed data (unlike HGB, which handles NaN natively).
        "feature_medians": feature_medians,
        # If set, inference-time code MUST append one {feature}_was_missing
        # column (1.0/0.0) per listed feature, in this order, AFTER imputing —
        # so the model can distinguish "truly average" from "was missing".
        "missing_indicator_features": missing_indicator_features,
    }
    manifest_path = output_dir / f"tuned_{model_key}_speed_model_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Tuned %s model + manifest saved → %s", model_key, model_path)
    return model, y_pred, best_params


def main(
    data_dirs: list[Path],
    output_dir: Path | None = None,
    tune_hgb: bool = True,
    tune_n_iter: int = 25,
    tune_cv_splits: int = 4,
    tuned_model_keys: list[str] | None = None,
    tuned_fixed_params: dict[str, dict[str, HyperParamValue]] | None = None,
) -> None:
    # --- Determine output directory -------------------------------------------
    if output_dir is None:
        output_dir = data_dirs[0] if len(data_dirs) == 1 else Path("reports/realtime")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load, clean, outlier-filter from all agencies ------------------------
    all_dfs: list[pd.DataFrame] = []
    agency_names: list[str] = []
    for data_dir in data_dirs:
        agency_label = data_dir.name
        agency_names.append(agency_label)
        csv_path = _resolve_input(data_dir)
        df_agency = load_and_clean(csv_path, agency_label=agency_label)
        df_agency = remove_outliers(df_agency)
        df_agency = add_temporal_features(df_agency)
        all_dfs.append(df_agency)

    df = pd.concat(all_dfs, ignore_index=True)
    log.info(
        "Combined data from %d agencies (%s): %d rows",
        len(data_dirs),
        ", ".join(agency_names),
        len(df),
    )

    fit_and_evaluate_models(
        df,
        agency_names,
        output_dir,
        tune_hgb=tune_hgb,
        tune_n_iter=tune_n_iter,
        tune_cv_splits=tune_cv_splits,
        tuned_model_keys=tuned_model_keys,
        tuned_fixed_params=tuned_fixed_params,
    )


def fit_and_evaluate_models(
    df: pd.DataFrame,
    agency_names: list[str],
    output_dir: Path,
    tune_hgb: bool = True,
    tune_n_iter: int = 25,
    tune_cv_splits: int = 4,
    tuned_model_keys: list[str] | None = None,
    tuned_fixed_params: dict[str, dict[str, HyperParamValue]] | None = None,
) -> pd.DataFrame | None:
    """Aggregate cleaned per-trip link speeds, fit models, and save all outputs.

    *df* must already be cleaned (``load_and_clean`` / ``remove_outliers`` /
    ``add_temporal_features`` applied) with one row per trip-link observation.
    Shared by both the JSONL-scrape pipeline (``main`` below) and the gtfsrt.io
    archive pipeline (``fit_archive_speed_models.py``), which differ only in how
    they load and clean the raw per-trip data.

    If *tune_hgb*, also fits the tuned models in ``TUNED_MODEL_CONFIGS`` (by
    default all of them: ``hgb``, ``rf``, ``gbr``; pass *tuned_model_keys* to
    fit only a subset, e.g. ``["rf"]``) using a coarser FHWA-style
    ``functional_class`` feature (in place of raw ``highway``) and a spatial
    (``GroupKFold`` by road_id) hyperparameter search — the feature-engineering
    and tuning steps recommended in the model summary.

    Returns the results DataFrame, or ``None`` if there was too little data to
    train on.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Aggregate to (road x hour x weekday/weekend) -------------------------
    agg = aggregate_to_road_hour(df)
    agg_target = "mph_moving_mean"

    # Require ≥3 trips contributing to each aggregated observation
    agg = agg[agg["n_trips"] >= 3].copy()
    log.info("After min-trip filter: %d aggregated rows", len(agg))

    if len(agg) < 50:
        log.error("Too few rows (%d) — cannot train models.", len(agg))
        return None

    # --- Build features -------------------------------------------------------
    y = agg[agg_target].values
    sample_weights = agg["total_observations"].values.astype(float)

    X_full, encoder = build_feature_matrix(agg, fit=True)

    num_cols = NUMERIC_FEATURES + TEMPORAL_FEATURES
    cat_names = list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = num_cols + cat_names

    log.info(
        "Feature matrix: %d samples x %d features  (target: %s)",
        X_full.shape[0],
        X_full.shape[1],
        agg_target,
    )

    # --- Spatial train/test split (by road_id) --------------------------------
    # Holding out entire roads simulates predicting speeds on unseen roads,
    # which is the real deployment scenario for new agencies.
    groups = agg["road_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_full, y, groups=groups))

    X_train, X_test = X_full[train_idx], X_full[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    w_train, w_test = sample_weights[train_idx], sample_weights[test_idx]

    n_roads_train = len(set(groups[train_idx]))
    n_roads_test = len(set(groups[test_idx]))
    log.info(
        "Spatial split: %d train (%d roads)  /  %d test (%d roads, held-out)",
        len(y_train),
        n_roads_train,
        len(y_test),
        n_roads_test,
    )

    # Per-agency test breakdown
    if "agency" in agg.columns:
        for ag in sorted(agg["agency"].unique()):
            ag_test_mask = agg.iloc[test_idx]["agency"].values == ag
            n_ag = ag_test_mask.sum()
            log.info("  Test set — %s: %d rows", ag, n_ag)

    # Impute NaN with training-set medians (for LR and RF)
    col_medians = np.nanmedian(X_train, axis=0)
    X_train_imp = np.where(np.isnan(X_train), col_medians, X_train)
    X_test_imp = np.where(np.isnan(X_test), col_medians, X_test)

    results: list[ModelMetrics] = []

    # --- 0. Baseline: speed_limit x constant ----------------------------------
    log.info("Fitting Speed-Limit Baseline (speed = maxspeed x k) …")
    speed_limit_idx = NUMERIC_FEATURES.index("maxspeed_mph")
    sl_train = X_train_imp[:, speed_limit_idx]
    sl_test = X_test_imp[:, speed_limit_idx]

    sl_known_mask = sl_train > 0
    if sl_known_mask.sum() > 0:
        w_known = w_train[sl_known_mask]
        k_opt = np.sum(
            w_known * y_train[sl_known_mask] * sl_train[sl_known_mask]
        ) / np.sum(w_known * sl_train[sl_known_mask] ** 2)
    else:
        k_opt = 0.6

    fallback_speed = np.average(y_train, weights=w_train)
    log.info(
        "  Optimal k = %.4f   (fallback for missing speed limit = %.1f mph)",
        k_opt,
        fallback_speed,
    )

    y_pred_baseline = np.where(sl_test > 0, sl_test * k_opt, fallback_speed)
    results.append(
        evaluate_model("Baseline (speed_limit x k)", y_test, y_pred_baseline, w_test)
    )

    # --- 1. Linear Regression -------------------------------------------------
    log.info("Fitting Linear Regression …")
    lr = LinearRegression()
    lr.fit(X_train_imp, y_train, sample_weight=w_train)
    y_pred_lr = lr.predict(X_test_imp)
    results.append(evaluate_model("Linear Regression", y_test, y_pred_lr, w_test))

    coef_idx = np.argsort(np.abs(lr.coef_))[::-1]
    log.info("  Top LR coefficients:")
    for i in coef_idx[:5]:
        log.info("    %-30s  %+.4f", all_feature_names[i], lr.coef_[i])

    # --- 2. Random Forest -----------------------------------------------------
    log.info("Fitting Random Forest …")
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train_imp, y_train, sample_weight=w_train)
    y_pred_rf = rf.predict(X_test_imp)
    results.append(evaluate_model("Random Forest", y_test, y_pred_rf, w_test))

    fi = pd.Series(rf.feature_importances_, index=all_feature_names).sort_values(
        ascending=False
    )
    log.info("  Top RF feature importances:")
    for feat, imp in fi.head(8).items():
        log.info("    %-30s  %.4f", feat, imp)

    # --- 3. Histogram Gradient Boosting ---------------------------------------
    log.info("Fitting Histogram Gradient Boosting …")
    hgb = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=10,
        random_state=42,
    )
    hgb.fit(X_train, y_train, sample_weight=w_train)
    y_pred_hgb = hgb.predict(X_test)
    results.append(
        evaluate_model("Histogram Gradient Boosting", y_test, y_pred_hgb, w_test)
    )

    # --- 4. Tuned models (feature engineering + spatial hyperparameter CV) ----
    # Feature engineering: coarsen the raw OSM `highway` tag (~15 subtypes,
    # some very sparse) to a small FHWA-style functional_class hierarchy that's
    # more consistent across agencies with different OSM tagging conventions.
    # Tuning: spatial (GroupKFold by road_id) search so hyperparameters are
    # chosen for generalization to unseen roads, matching the final eval split.
    tuned_models: dict[str, Any] = {}
    tuned_best_params: dict[str, dict[str, HyperParamValue]] = {}
    tuned_cat_names: list[str] = []
    tuned_train_matrices: dict[str, np.ndarray] = {}
    tuned_test_matrices: dict[str, np.ndarray] = {}
    if tune_hgb:
        agg["functional_class"] = (
            agg["highway"]
            .map(HIGHWAY_TO_FUNCTIONAL_CLASS)
            .fillna(FUNCTIONAL_CLASS_DEFAULT)
        )
        X_tuned_full, tuned_encoder = build_feature_matrix(
            agg, categorical_features=["functional_class"], fit=True
        )
        tuned_cat_names = list(
            tuned_encoder.get_feature_names_out(["functional_class"])
        )
        X_tuned_train, X_tuned_test = X_tuned_full[train_idx], X_tuned_full[test_idx]
        # RF and classic GBR don't support NaN natively (unlike HGB), so build
        # a train-median-imputed variant for them, mirroring the imputation
        # used for the untuned LR/RF models above.
        tuned_col_medians = np.nanmedian(X_tuned_train, axis=0)
        X_tuned_train_imp = np.where(
            np.isnan(X_tuned_train), tuned_col_medians, X_tuned_train
        )
        X_tuned_test_imp = np.where(
            np.isnan(X_tuned_test), tuned_col_medians, X_tuned_test
        )
        # Append {feature}_was_missing indicator columns for RF/GBR so the
        # median fill isn't silently treated as a real observed value — this
        # preserves all rows (dropping is not viable: maxspeed_mph/lanes are
        # missing on ~70% of OSM road segments) while still letting the model
        # learn from the missingness signal itself.
        numeric_missing_train = np.isnan(
            X_tuned_train[:, : len(NUMERIC_FEATURES)]
        ).astype(np.float32)
        numeric_missing_test = np.isnan(
            X_tuned_test[:, : len(NUMERIC_FEATURES)]
        ).astype(np.float32)
        X_tuned_train_imp_ind = np.hstack([X_tuned_train_imp, numeric_missing_train])
        X_tuned_test_imp_ind = np.hstack([X_tuned_test_imp, numeric_missing_test])
        tuned_input_order = NUMERIC_FEATURES + TEMPORAL_FEATURES + tuned_cat_names
        tuned_feature_medians = dict(zip(tuned_input_order, tuned_col_medians.tolist()))

        selected_keys = tuned_model_keys or list(TUNED_MODEL_CONFIGS)
        for model_key, config in TUNED_MODEL_CONFIGS.items():
            if model_key not in selected_keys:
                continue
            if config.get("needs_imputation"):
                X_tr, X_te = X_tuned_train_imp_ind, X_tuned_test_imp_ind
            else:
                X_tr, X_te = X_tuned_train, X_tuned_test
            model, _, best_params = _tune_and_persist_model(
                model_key,
                config,
                X_tr,
                X_te,
                y_train,
                y_test,
                w_train,
                w_test,
                groups[train_idx],
                tune_n_iter,
                tune_cv_splits,
                tuned_encoder,
                tuned_cat_names,
                output_dir,
                results,
                feature_medians=tuned_feature_medians
                if config.get("needs_imputation")
                else None,
                missing_indicator_features=NUMERIC_FEATURES
                if config.get("needs_imputation")
                else None,
                fixed_params=(tuned_fixed_params or {}).get(model_key),
            )
            tuned_models[model_key] = model
            tuned_best_params[model_key] = best_params
            tuned_train_matrices[model_key] = X_tr
            tuned_test_matrices[model_key] = X_te
    else:
        log.info("Skipping tuned models (tune_hgb=False)")

    # --- Results summary table ------------------------------------------------
    results_df = pd.DataFrame(results)
    log.info("\n%s", results_df.to_string(index=False))

    # --- Save outputs ---------------------------------------------------------
    results_path = output_dir / "speed_model_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Model metrics saved → %s", results_path)

    fi_path = output_dir / "feature_importances.csv"
    fi.to_csv(fi_path, header=["importance"])
    log.info("Feature importances saved → %s", fi_path)

    agg_path = output_dir / "aggregated_training_data.csv"
    agg.to_csv(agg_path, index=False)
    log.info("Aggregated training data saved → %s", agg_path)

    # --- Save test-set predictions (used by visualize_speed_models.py) --------
    test_meta_cols = [
        "road_id",
        "hour",
        "is_weekday",
        "highway",
        "maxspeed_mph",
        "lanes",
        "grade",
        "grade_abs",
        "link_length_km",
        "n_trips",
    ]
    if "agency" in agg.columns:
        test_meta_cols = ["agency"] + test_meta_cols
    for col in ["n_stops", "scheduled_speed_mph", "functional_class"]:
        if col in agg.columns:
            test_meta_cols.append(col)
    test_df = agg.iloc[test_idx][test_meta_cols].copy().reset_index(drop=True)
    test_df["actual_mph"] = y_test
    test_df["pred_baseline"] = y_pred_baseline
    test_df["pred_lr"] = y_pred_lr
    test_df["pred_rf"] = y_pred_rf
    test_df["pred_hgb"] = y_pred_hgb
    for model_key, model in tuned_models.items():
        test_df[f"pred_{model_key}_tuned"] = model.predict(
            tuned_test_matrices[model_key]
        )
    test_preds_path = output_dir / "test_predictions.csv"
    test_df.to_csv(test_preds_path, index=False)
    log.info("Test predictions saved → %s", test_preds_path)

    # --- Save predictions for ALL rows (train + test) for full-network map ---
    # Compute HGB predictions on training rows so every road can be visualized.
    y_pred_hgb_train = hgb.predict(X_train)
    y_pred_baseline_train = np.where(sl_train > 0, sl_train * k_opt, fallback_speed)
    y_pred_lr_train = lr.predict(X_train_imp)
    y_pred_rf_train = rf.predict(X_train_imp)

    train_df = agg.iloc[train_idx][test_meta_cols].copy().reset_index(drop=True)
    train_df["actual_mph"] = y_train
    train_df["pred_baseline"] = y_pred_baseline_train
    train_df["pred_lr"] = y_pred_lr_train
    train_df["pred_rf"] = y_pred_rf_train
    train_df["pred_hgb"] = y_pred_hgb_train
    for model_key, model in tuned_models.items():
        train_df[f"pred_{model_key}_tuned"] = model.predict(
            tuned_train_matrices[model_key]
        )

    test_df["split"] = "test"
    train_df["split"] = "train"
    all_preds_path = output_dir / "all_predictions.csv"
    pd.concat([train_df, test_df], ignore_index=True).to_csv(
        all_preds_path, index=False
    )
    log.info("All predictions saved → %s", all_preds_path)

    # Also record train-set road IDs (for backward-compat context layer on the folium map)
    train_roads_path = output_dir / "train_road_ids.csv"
    pd.Series(sorted(set(groups[train_idx])), name="road_id").to_csv(
        train_roads_path, index=False
    )
    log.info("Train road IDs saved  → %s", train_roads_path)

    # --- Detailed summary report ----------------------------------------------
    best = max(results, key=lambda r: r["r2"])
    worst = min(results, key=lambda r: r["r2"])

    agencies_str = ", ".join(agency_names)
    summary_path = output_dir / "speed_model_summary.md"
    summary = textwrap.dedent(f"""\
    # Transit Speed Prediction — Model Summary

    ## Goal
    Predict average transit bus moving-speed on any US road segment given only
    OSM attributes and time-of-day context.  The model should generalize to
    agencies without realtime data by relying on universally available features.

    ## Dataset
    - **Agencies**: {agencies_str}
    - **Per-trip rows (after cleaning + outlier removal)**: {len(df):,}
    - **Aggregated to (road x hour x weekday/weekend)**: {len(agg):,} groups (≥3 trips each)
    - **Target variable**: `{agg_target}` — observation-weighted mean moving speed (mph)
    - **Outlier removal**: IQR per road + hard [{SPEED_FLOOR_MPH}–{SPEED_CEIL_MPH}] mph
    - **Train / Test split**: spatial hold-out by road_id — {len(y_train):,} train ({n_roads_train} roads) / {len(y_test):,} test ({n_roads_test} held-out roads)

    ## Features (all OSM-derivable or temporal)
    | Category | Features |
    |----------|----------|
    | Road attributes | `maxspeed_mph`, `lanes`, `grade`, `grade_abs`, `link_length_km` |
    | Road type | `highway` (one-hot: {", ".join(cat_names)}) |
    | Road type (tuned model) | `functional_class` (one-hot: {", ".join(tuned_cat_names) if tuned_cat_names else "n/a"}) |
    | Temporal | `hour`, `is_weekday`, `is_peak` |

    ## Model Performance

    | Model | R² | RMSE (mph) | MAE (mph) |
    |-------|---:|----------:|----------:|
    """)

    for r in results:
        summary += f"| {r['model']} | {r['r2']:.4f} | {r['rmse_mph']:.2f} | {r['mae_mph']:.2f} |\n"

    summary += textwrap.dedent(f"""
    ## Key Findings

    - **Best model**: {best["model"]} (R² = {best["r2"]:.4f}, RMSE = {best["rmse_mph"]:.2f} mph)
    - **Worst model**: {worst["model"]} (R² = {worst["r2"]:.4f}, RMSE = {worst["rmse_mph"]:.2f} mph)
    - The spatial hold-out split (entire roads held out) is deliberately harder
      than random splitting and better reflects real generalization to new
      agencies/cities where no realtime data exists.
    """)

    if tune_hgb and tuned_models:
        functional_classes = ", ".join(
            dict.fromkeys(HIGHWAY_TO_FUNCTIONAL_CLASS.values())
        )
        summary += textwrap.dedent(f"""
        ### Hyperparameter Tuning & Feature Engineering

        - **Feature engineering**: raw `highway` (one-hot, {len(cat_names)} levels)
          replaced with a coarser FHWA-style `functional_class`
          ({len(tuned_cat_names)} levels: {functional_classes}) to reduce
          sparsity in rare highway subtypes and normalize across agencies'
          differing OSM tagging conventions.
        - **Hyperparameter search**: `RandomizedSearchCV` ({tune_n_iter} samples,
          `GroupKFold` by road_id, {tune_cv_splits} folds), per model \u2014 see
          each model's search space in `TUNED_MODEL_CONFIGS`.
        """)
        untuned_labels = {
            "hgb": "Histogram Gradient Boosting",
            "rf": "Random Forest",
        }
        for model_key, config in TUNED_MODEL_CONFIGS.items():
            if model_key not in tuned_models:
                continue
            tuned_r2 = next(r["r2"] for r in results if r["model"] == config["label"])
            untuned_label = untuned_labels.get(model_key)
            untuned_r2 = next(
                (r["r2"] for r in results if r["model"] == untuned_label),
                None,
            )
            comparison = (
                f"R\u00b2 {untuned_r2:.4f} \u2192 {tuned_r2:.4f}"
                if untuned_r2 is not None
                else f"R\u00b2 {tuned_r2:.4f} (no untuned baseline fit)"
            )
            summary += (
                f"- **{config['label']}**: best params "
                f"`{tuned_best_params[model_key]}` \u2014 {comparison}\n"
            )

    summary += textwrap.dedent("""
    ### Top Features (Random Forest importance)
    """)
    for feat, imp in fi.head(8).items():
        summary += f"- `{feat}`: {imp:.4f}\n"

    summary += textwrap.dedent("""
    ## Recommendations for Improvement

    1. **More agencies**: Add data from agencies in different city sizes,
       climates, and traffic patterns to learn generalizable speed–road
       relationships.
    2. **Spatial features**: Add intersection density or traffic-signal density
       within a buffer — these are computable from OSM for any US city and
       strongly affect transit speeds.
    3. **Better temporal bucketing**: Experiment with finer (30-min) or coarser
       (AM peak / midday / PM peak / evening) time bins depending on data volume.
    4. **Separate dwell model**: Predict dwell time independently and combine with
       moving-speed model for total link traversal time.
    5. **Target engineering**: Try log(speed) to reduce right-skew, or quantile
       regression to predict speed distributions rather than point estimates.
    6. **Seasonal / weather effects**: Incorporate month or temperature once data
       spans multiple seasons.
    7. **Hyperparameter tuning**: Use cross-validated search (spatial CV with
       GroupKFold) — current parameters are reasonable defaults.
    8. **Functional class mapping**: Map OSM `highway` tags to FHWA functional
       classes for a more standardized, coarser road hierarchy that's consistent
       across OSM tagging conventions in different regions.
    """)

    with open(summary_path, "w") as f:
        f.write(summary)
    log.info("Summary report saved → %s", summary_path)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit speed prediction models")
    parser.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        dest="data_dirs",
        help="Agency data directory (may be repeated for multi-agency training)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for model outputs (default: first data-dir, or reports/realtime for multi-agency)",
    )
    parser.add_argument(
        "--no-tune-hgb",
        dest="tune_hgb",
        action="store_false",
        help="Skip the tuned HGB model (feature engineering + hyperparameter search)",
    )
    parser.add_argument(
        "--tune-n-iter",
        type=int,
        default=25,
        help="Number of RandomizedSearchCV samples for HGB tuning (default: 25)",
    )
    parser.add_argument(
        "--tune-cv-splits",
        type=int,
        default=4,
        help="Number of spatial GroupKFold splits for HGB tuning (default: 4)",
    )
    parser.add_argument(
        "--tuned-models",
        default=None,
        help="Comma-separated subset of tuned models to fit: hgb,rf,gbr "
        "(default: all three)",
    )
    parser.add_argument(
        "--fixed-params-json",
        default=None,
        help="JSON dict of {model_key: {param: value}} to skip the "
        "RandomizedSearchCV step and refit directly with known params, e.g. "
        '\'{"rf": {"n_estimators": 300, "max_depth": 8}}\'',
    )
    args = parser.parse_args()
    dirs = args.data_dirs if args.data_dirs else DEFAULT_DATA_DIRS
    main(
        dirs,
        args.output_dir,
        tune_hgb=args.tune_hgb,
        tune_n_iter=args.tune_n_iter,
        tune_cv_splits=args.tune_cv_splits,
        tuned_model_keys=args.tuned_models.split(",") if args.tuned_models else None,
        tuned_fixed_params=json.loads(args.fixed_params_json)
        if args.fixed_params_json
        else None,
    )
