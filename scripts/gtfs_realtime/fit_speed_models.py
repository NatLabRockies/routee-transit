"""Fit speed prediction models to aggregated realtime link speed data.

Reads the combined per-trip link speed CSV produced by
``aggregate_portland_me.py``, removes outliers, aggregates to mean speed per
road link × time-of-day bin × weekday/weekend, and trains three regression
models using only features available from OSM (generalizable to any US
transit agency).

1. Linear Regression (OLS baseline)
2. Random Forest
3. Histogram Gradient Boosting (handles NaN natively)

The train/test split is spatial — entire road segments are held out — so
that metrics reflect the real use-case of predicting speeds on *unseen*
roads.

Usage
-----
    python fit_speed_models.py [--data-dir PATH] [--input FILE]
"""

import argparse
import json
import logging
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("reports/realtime/greater_portland_me")

# Only features derivable from OSM — no agency- or trip-specific data.
NUMERIC_FEATURES = ["maxspeed_mph", "lanes", "grade", "grade_abs", "link_length_km"]
CATEGORICAL_FEATURES = ["highway"]
TEMPORAL_FEATURES = ["hour", "is_weekday", "is_peak"]
TARGET = "mph_moving"

# Outlier thresholds
SPEED_FLOOR_MPH = 1.0  # below this is likely GPS noise / dwell misattribution
SPEED_CEIL_MPH = 65.0  # above this is implausible for transit buses
IQR_MULTIPLIER = 1.5  # per-road IQR fence


def _resolve_input(data_dir: Path, input_file: str | None) -> Path:
    """Find the best available per-trip speed CSV."""
    if input_file:
        return Path(input_file)
    all_days = data_dir / "realtime_link_speeds_all_days.csv"
    if all_days.exists():
        return all_days
    candidates = sorted(data_dir.glob("realtime_link_speeds_2*.csv"))
    candidates = [f for f in candidates if "aggregated" not in f.name]
    if not candidates:
        raise FileNotFoundError(f"No per-trip speed CSVs found in {data_dir}")
    return candidates[-1]


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Load per-trip link speeds and apply basic sanity filters."""
    log.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)
    log.info("Raw rows: %d", len(df))

    df = df.dropna(subset=[TARGET])
    df = df[np.isfinite(df[TARGET])]
    # Keep only directly observed links (≥2 GPS pings on the link)
    if "speed_source" in df.columns:
        df = df[df["speed_source"] == "observed"]
    # Hard floor/ceiling
    df = df[(df[TARGET] >= SPEED_FLOOR_MPH) & (df[TARGET] <= SPEED_CEIL_MPH)]
    log.info("After basic filters: %d rows", len(df))
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove per-road outliers using the IQR method.

    For each road_id, speeds outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR] are
    removed. Roads with fewer than 4 observations skip IQR filtering (the
    hard floor/ceiling is still in effect).
    """
    n_before = len(df)
    keep_mask = pd.Series(True, index=df.index)

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
        n_removed, 100 * n_removed / max(n_before, 1), len(df),
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
    # Carry forward road-level attributes (constant per road_id)
    road_attrs = ["highway", "maxspeed_mph", "lanes", "grade", "grade_abs", "link_length_km"]
    first_cols = {c: "first" for c in road_attrs if c in df.columns}

    def _weighted_mean(g: pd.DataFrame) -> pd.Series:
        w = g["n_observations"].values.astype(float)
        total_w = w.sum()
        if total_w == 0:
            w = np.ones(len(g))
            total_w = float(len(g))
        return pd.Series({
            "mph_moving_mean": np.average(g[TARGET].values, weights=w),
            "mph_moving_std": g[TARGET].std(),
            "n_trips": len(g),
            "total_observations": int(total_w),
        })

    agg = df.groupby(group_cols).apply(_weighted_mean, include_groups=False).reset_index()

    # Attach road attributes from first occurrence
    road_props = df.groupby("road_id")[list(first_cols.keys())].first().reset_index()
    agg = agg.merge(road_props, on="road_id", how="left")

    # is_peak is derivable from hour
    agg["is_peak"] = agg["hour"].apply(
        lambda h: 1 if (7 <= h <= 9 or 16 <= h <= 18) else 0
    )

    log.info(
        "Aggregated to %d (road × hour × weekday/weekend) groups from %d trips",
        len(agg), df["trip_id"].nunique() if "trip_id" in df.columns else -1,
    )
    return agg


def build_feature_matrix(
    df: pd.DataFrame,
    encoder: OneHotEncoder | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, OneHotEncoder]:
    """Build the feature matrix X from an aggregated DataFrame."""
    num_cols = NUMERIC_FEATURES + TEMPORAL_FEATURES
    X_num = df[num_cols].values.astype(float)

    cat_data = df[CATEGORICAL_FEATURES].fillna("unknown").values
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    if fit:
        X_cat = encoder.fit_transform(cat_data)
    else:
        X_cat = encoder.transform(cat_data)

    X = np.hstack([X_num, X_cat])
    return X, encoder


def evaluate_model(
    name: str, y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None,
) -> dict:
    """Compute regression metrics (optionally observation-weighted)."""
    r2 = r2_score(y_true, y_pred, sample_weight=weights)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=weights))
    mae = mean_absolute_error(y_true, y_pred, sample_weight=weights)
    log.info("  %-35s  R²=%.4f  RMSE=%.2f mph  MAE=%.2f mph", name, r2, rmse, mae)
    return {"model": name, "r2": r2, "rmse_mph": rmse, "mae_mph": mae}


def main(data_dir: Path, input_file: str | None = None) -> None:
    csv_path = _resolve_input(data_dir, input_file)

    # --- Load, clean, outlier-filter ------------------------------------------
    df = load_and_clean(csv_path)
    df = remove_outliers(df)
    df = add_temporal_features(df)

    # --- Aggregate to (road × hour × weekday/weekend) -------------------------
    agg = aggregate_to_road_hour(df)
    agg_target = "mph_moving_mean"

    # Require ≥3 trips contributing to each aggregated observation
    agg = agg[agg["n_trips"] >= 3].copy()
    log.info("After min-trip filter: %d aggregated rows", len(agg))

    if len(agg) < 50:
        log.error("Too few rows (%d) — cannot train models.", len(agg))
        return

    # --- Build features -------------------------------------------------------
    y = agg[agg_target].values
    sample_weights = agg["total_observations"].values.astype(float)

    X_full, encoder = build_feature_matrix(agg, fit=True)

    num_cols = NUMERIC_FEATURES + TEMPORAL_FEATURES
    cat_names = list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = num_cols + cat_names

    log.info(
        "Feature matrix: %d samples × %d features  (target: %s)",
        X_full.shape[0], X_full.shape[1], agg_target,
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
        len(y_train), n_roads_train, len(y_test), n_roads_test,
    )

    # Impute NaN with training-set medians (for LR and RF)
    col_medians = np.nanmedian(X_train, axis=0)
    X_train_imp = np.where(np.isnan(X_train), col_medians, X_train)
    X_test_imp = np.where(np.isnan(X_test), col_medians, X_test)

    results: list[dict] = []

    # --- 0. Baseline: speed_limit × constant ----------------------------------
    # Fit a single scalar k such that predicted_speed = maxspeed_mph × k.
    # On roads with no speed limit, fall back to the training-set mean speed.
    log.info("Fitting Speed-Limit Baseline (speed = maxspeed × k) …")
    speed_limit_idx = NUMERIC_FEATURES.index("maxspeed_mph")
    sl_train = X_train_imp[:, speed_limit_idx]
    sl_test = X_test_imp[:, speed_limit_idx]

    # Only fit k on rows where speed limit is known (non-zero, non-NaN)
    sl_known_mask = sl_train > 0
    if sl_known_mask.sum() > 0:
        # Weighted least-squares for k: minimise Σ w·(y − k·sl)²  → k = Σ(w·y·sl) / Σ(w·sl²)
        w_known = w_train[sl_known_mask]
        k_opt = (
            np.sum(w_known * y_train[sl_known_mask] * sl_train[sl_known_mask])
            / np.sum(w_known * sl_train[sl_known_mask] ** 2)
        )
    else:
        k_opt = 0.6  # sensible default

    fallback_speed = np.average(y_train, weights=w_train)
    log.info("  Optimal k = %.4f   (fallback for missing speed limit = %.1f mph)", k_opt, fallback_speed)

    y_pred_baseline = np.where(sl_test > 0, sl_test * k_opt, fallback_speed)
    results.append(evaluate_model("Baseline (speed_limit × k)", y_test, y_pred_baseline, w_test))

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
    results.append(evaluate_model("Histogram Gradient Boosting", y_test, y_pred_hgb, w_test))

    # --- Results summary table ------------------------------------------------
    results_df = pd.DataFrame(results)
    log.info("\n%s", results_df.to_string(index=False))

    # --- Save outputs ---------------------------------------------------------
    results_path = data_dir / "speed_model_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Model metrics saved → %s", results_path)

    fi_path = data_dir / "feature_importances.csv"
    fi.to_csv(fi_path, header=["importance"])
    log.info("Feature importances saved → %s", fi_path)

    agg_path = data_dir / "aggregated_training_data.csv"
    agg.to_csv(agg_path, index=False)
    log.info("Aggregated training data saved → %s", agg_path)

    # --- Detailed summary report ----------------------------------------------
    best = max(results, key=lambda r: r["r2"])
    worst = min(results, key=lambda r: r["r2"])

    n_raw = pd.read_csv(csv_path, usecols=[TARGET], nrows=0)  # just for the path
    summary_path = data_dir / "speed_model_summary.md"
    summary = textwrap.dedent(f"""\
    # Portland ME — Transit Speed Prediction Model Summary

    ## Goal
    Predict average transit bus moving-speed on any US road segment given only
    OSM attributes and time-of-day context.  The model should generalize to
    agencies without realtime data by relying on universally available features.

    ## Dataset
    - **Source**: Greater Portland METRO GTFS-RT link-level speed observations
    - **Per-trip rows loaded**: {len(df):,}
    - **After outlier removal (IQR per road + hard [{SPEED_FLOOR_MPH}–{SPEED_CEIL_MPH}] mph)**: {len(df):,}
    - **Aggregated to (road × hour × weekday/weekend)**: {len(agg):,} groups (≥3 trips each)
    - **Target variable**: `{agg_target}` — observation-weighted mean moving speed (mph)
    - **Train / Test split**: spatial hold-out by road_id — {len(y_train):,} train ({n_roads_train} roads) / {len(y_test):,} test ({n_roads_test} held-out roads)

    ## Features (all OSM-derivable or temporal)
    | Category | Features |
    |----------|----------|
    | Road attributes | `maxspeed_mph`, `lanes`, `grade`, `grade_abs`, `link_length_km` |
    | Road type | `highway` (one-hot: {', '.join(cat_names)}) |
    | Temporal | `hour`, `is_weekday`, `is_peak` |

    ## Model Performance

    | Model | R² | RMSE (mph) | MAE (mph) |
    |-------|---:|----------:|----------:|
    """)

    for r in results:
        summary += f"| {r['model']} | {r['r2']:.4f} | {r['rmse_mph']:.2f} | {r['mae_mph']:.2f} |\n"

    summary += textwrap.dedent(f"""
    ## Key Findings

    - **Best model**: {best['model']} (R² = {best['r2']:.4f}, RMSE = {best['rmse_mph']:.2f} mph)
    - **Worst model**: {worst['model']} (R² = {worst['r2']:.4f}, RMSE = {worst['rmse_mph']:.2f} mph)
    - The spatial hold-out split (entire roads held out) is deliberately harder
      than random splitting and better reflects real generalization to new
      agencies/cities where no realtime data exists.

    ### Top Features (Random Forest importance)
    """)
    for feat, imp in fi.head(8).items():
        summary += f"- `{feat}`: {imp:.4f}\n"

    summary += textwrap.dedent("""
    ## Recommendations for Improvement

    1. **Multi-agency training**: Add data from agencies in different city sizes,
       climates, and traffic patterns (King County, CDTA Albany, etc.) to learn
       generalizable speed–road relationships rather than Portland-specific ones.
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit speed prediction models")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory with per-trip speed CSVs",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Specific CSV file to use as input",
    )
    args = parser.parse_args()
    main(args.data_dir, args.input)
