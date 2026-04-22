"""Visualize speed prediction model diagnostics.

Loads test-set predictions saved by ``fit_speed_models.py`` and produces:

1. Folium error map (``error_map.html``)
2. Predicted-vs-actual scatter (``predicted_vs_actual.png``)
3. Accuracy by speed range (``accuracy_by_speed_range.png``)
4. Residuals by hour (``residuals_by_hour.png``)
5. Residuals by highway type (``residuals_by_highway.png``)
6. Feature importances bar chart (``feature_importances.png``)
7. Residual-vs-feature diagnostic grid (``residuals_vs_features.png``)

Usage
-----
    # Must run fit_speed_models.py first to generate test_predictions.csv
    python visualize_speed_models.py \\
        --data-dir reports/realtime/greater_portland_me \\
        --data-dir reports/realtime/cdta_albany
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shapely
from branca.colormap import linear
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
log = logging.getLogger(__name__)

MODEL_COLS = {
    "Baseline": "pred_baseline",
    "Linear Regression": "pred_lr",
    "Random Forest": "pred_rf",
    "Hist. Gradient Boost": "pred_hgb",
}
PRIMARY_MODEL = "pred_hgb"
PRIMARY_LABEL = "Hist. Gradient Boost"

DEFAULT_DATA_DIRS = [
    Path("reports/realtime/greater_portland_me"),
    Path("reports/realtime/cdta_albany"),
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _load_geometry_lookup(data_dirs: list[Path], multi_agency: bool) -> dict[str, str]:
    """Build road_id → WKT geometry dict from per-agency aggregated CSVs."""
    lookup: dict[str, str] = {}
    for data_dir in data_dirs:
        agg_csv = data_dir / "realtime_link_speeds_aggregated_all_days.csv"
        all_days = data_dir / "realtime_link_speeds_all_days.csv"

        if agg_csv.exists():
            df = pd.read_csv(agg_csv, usecols=["road_id", "geom"])
        elif all_days.exists():
            df = pd.read_csv(all_days, usecols=["road_id", "geom"])
            df = df.drop_duplicates(subset=["road_id"])
        else:
            log.warning("No geometry CSV found in %s — skipping", data_dir)
            continue

        df = df.dropna(subset=["geom"])
        agency_label = data_dir.name
        for _, row in df.iterrows():
            rid = row["road_id"]
            key = f"{agency_label}_{rid}" if multi_agency else str(rid)
            lookup[key] = row["geom"]

    log.info("Geometry lookup: %d roads with geometry", len(lookup))
    return lookup


# ---------------------------------------------------------------------------
# Plot 1 — Folium error map
# ---------------------------------------------------------------------------


def _aggregate_roads(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate predictions to per-road level (mean across hours/days)."""
    return (
        df.groupby("road_id")
        .agg(
            actual_mph=("actual_mph", "mean"),
            pred_hgb=(PRIMARY_MODEL, "mean"),
            highway=("highway", "first"),
            n_rows=("actual_mph", "size"),
        )
        .reset_index()
    )


def _attach_geometry(
    road_df: pd.DataFrame, geom_lookup: dict[str, str]
) -> gpd.GeoDataFrame | None:
    """Map road_id → WKT, parse geometry, return GeoDataFrame (or None)."""
    road_df = road_df.copy()
    road_df["geom_wkt"] = road_df["road_id"].map(geom_lookup)
    road_df = road_df.dropna(subset=["geom_wkt"])
    if road_df.empty:
        return None
    road_df["geometry"] = gpd.GeoSeries.from_wkt(road_df["geom_wkt"])
    return gpd.GeoDataFrame(road_df, geometry="geometry", crs="EPSG:4326")


def build_error_map(
    test_df: pd.DataFrame,
    train_road_ids: set[str],
    geom_lookup: dict[str, str],
    output_dir: Path,
    error_type: str = "abs",
    all_df: pd.DataFrame | None = None,
) -> None:
    """Interactive map with layers for error, predicted speed, and training context."""
    # Aggregate test predictions to per-road level
    road_agg = _aggregate_roads(test_df)
    road_agg["abs_err"] = (road_agg["pred_hgb"] - road_agg["actual_mph"]).abs()
    road_agg["pct_err"] = (
        100 * road_agg["abs_err"] / road_agg["actual_mph"].clip(lower=1)
    )
    road_agg["residual"] = road_agg["pred_hgb"] - road_agg["actual_mph"]

    gdf_test = _attach_geometry(road_agg, geom_lookup)
    if gdf_test is None:
        log.warning("No test roads have geometry — skipping error map")
        return

    # Build training-road GeoDataFrame for context
    train_geoms = {
        rid: geom_lookup[rid] for rid in train_road_ids if rid in geom_lookup
    }
    gdf_train = None
    if train_geoms:
        gdf_train = gpd.GeoDataFrame(
            {"road_id": list(train_geoms.keys())},
            geometry=gpd.GeoSeries.from_wkt(list(train_geoms.values())),
            crs="EPSG:4326",
        )

    # Build all-roads predicted-speed GeoDataFrame
    gdf_all = None
    if all_df is not None:
        all_road_agg = _aggregate_roads(all_df)
        split_per_road = all_df.groupby("road_id")["split"].first().reset_index()
        all_road_agg = all_road_agg.merge(split_per_road, on="road_id", how="left")
        gdf_all = _attach_geometry(all_road_agg, geom_lookup)

    # Map center
    centroid = gdf_test.union_all().centroid
    m = folium.Map(
        location=[centroid.y, centroid.x], zoom_start=12, tiles="cartodb-positron"
    )

    # --- Training roads layer (gray context) ---
    if gdf_train is not None:
        train_group = folium.FeatureGroup(name="Training roads (context)", show=True)
        for _, row in gdf_train.iterrows():
            coords = shapely.get_coordinates(row.geometry)
            points = [[coords[i, 1], coords[i, 0]] for i in range(len(coords))]
            folium.PolyLine(
                locations=points, color="#999999", weight=2, opacity=0.3
            ).add_to(train_group)
        train_group.add_to(m)

    # --- Test roads: prediction error layer ---
    err_col = "abs_err" if error_type == "abs" else "pct_err"
    vmax = 10.0 if error_type == "abs" else 50.0
    caption = "Absolute Error (mph)" if error_type == "abs" else "Percent Error (%)"
    colormap_err = linear.RdYlGn_09.scale(0, vmax)
    colormap_err = colormap_err.to_step(n=10)
    colormap_err.caption = caption
    color_scale_err = linear.RdYlGn_09.scale(0, vmax)

    test_group = folium.FeatureGroup(name=f"Test roads — {caption}", show=True)
    for _, row in gdf_test.iterrows():
        coords = shapely.get_coordinates(row.geometry)
        points = [[coords[i, 1], coords[i, 0]] for i in range(len(coords))]
        val = min(row[err_col], vmax)
        # Invert: low error → green (high end of RdYlGn)
        color = color_scale_err(vmax - val)
        popup_html = (
            f"<b>Road:</b> {row['road_id']}<br>"
            f"<b>Highway:</b> {row['highway']}<br>"
            f"<b>Actual:</b> {row['actual_mph']:.1f} mph<br>"
            f"<b>Predicted:</b> {row['pred_hgb']:.1f} mph<br>"
            f"<b>Abs Error:</b> {row['abs_err']:.1f} mph<br>"
            f"<b>Pct Error:</b> {row['pct_err']:.0f}%<br>"
            f"<b>Residual:</b> {row['residual']:+.1f} mph<br>"
            f"<b>N obs:</b> {row['n_rows']}"
        )
        folium.PolyLine(
            locations=points,
            popup=folium.Popup(popup_html, max_width=300),
            color=color,
            weight=4,
            opacity=0.8,
        ).add_to(test_group)
    test_group.add_to(m)
    m.add_child(colormap_err)

    # --- All roads: predicted speed layer ---
    if gdf_all is not None:
        speed_vmax = min(60.0, gdf_all["pred_hgb"].quantile(0.98))
        colormap_spd = linear.RdYlGn_09.scale(0, speed_vmax)
        colormap_spd = colormap_spd.to_step(n=10)
        colormap_spd.caption = "Predicted Speed (mph)"
        color_scale_spd = linear.RdYlGn_09.scale(0, speed_vmax)

        speed_group = folium.FeatureGroup(
            name="All roads — Predicted Speed", show=False
        )
        for _, row in gdf_all.iterrows():
            coords = shapely.get_coordinates(row.geometry)
            points = [[coords[i, 1], coords[i, 0]] for i in range(len(coords))]
            spd = max(0.0, min(row["pred_hgb"], speed_vmax))
            color = color_scale_spd(spd)
            split_label = row.get("split", "unknown")
            popup_html = (
                f"<b>Road:</b> {row['road_id']}<br>"
                f"<b>Highway:</b> {row['highway']}<br>"
                f"<b>Predicted:</b> {row['pred_hgb']:.1f} mph<br>"
                f"<b>Actual:</b> {row['actual_mph']:.1f} mph<br>"
                f"<b>Split:</b> {split_label}"
            )
            folium.PolyLine(
                locations=points,
                popup=folium.Popup(popup_html, max_width=300),
                color=color,
                weight=3,
                opacity=0.7,
            ).add_to(speed_group)
        speed_group.add_to(m)
        m.add_child(colormap_spd)

    folium.LayerControl().add_to(m)

    out = output_dir / "error_map.html"
    m.save(str(out))
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# Plot 2 — Predicted vs Actual scatter
# ---------------------------------------------------------------------------


def plot_predicted_vs_actual(test_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    max_speed = max(test_df["actual_mph"].max(), 50)

    for ax, (label, col) in zip(axes, MODEL_COLS.items()):
        y_true = test_df["actual_mph"].values
        y_pred = test_df[col].values
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        ax.scatter(y_true, y_pred, alpha=0.25, s=8, c="steelblue", edgecolors="none")
        ax.plot([0, max_speed], [0, max_speed], "k--", lw=1, label="1:1")
        ax.set_xlim(0, max_speed)
        ax.set_ylim(0, max_speed)
        ax.set_xlabel("Actual speed (mph)")
        ax.set_ylabel("Predicted speed (mph)")
        ax.set_title(f"{label}\nR²={r2:.3f}  MAE={mae:.1f} mph", fontsize=10)
        ax.set_aspect("equal")

    fig.suptitle("Predicted vs Actual Speed — All Models", fontsize=13, y=1.01)
    fig.tight_layout()
    out = output_dir / "predicted_vs_actual.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# Plot 3 — Accuracy by speed range
# ---------------------------------------------------------------------------


def plot_accuracy_by_speed_range(test_df: pd.DataFrame, output_dir: Path) -> None:
    bins = [0, 15, 20, 25, 30, 35, 40, 45, 100]
    labels = ["0–15", "15–20", "20–25", "25–30", "30–35", "35–40", "40–45", "45+"]
    test_df = test_df.copy()
    test_df = test_df.dropna(subset=["maxspeed_mph"])
    test_df = test_df[test_df["maxspeed_mph"] > 0]
    test_df["speed_bin"] = pd.cut(
        test_df["maxspeed_mph"], bins=bins, labels=labels, right=False
    )

    records = []
    for bin_label in labels:
        mask = test_df["speed_bin"] == bin_label
        n = mask.sum()
        if n == 0:
            continue
        for model_name, col in MODEL_COLS.items():
            subset = test_df.loc[mask]
            mae = (subset[col] - subset["actual_mph"]).abs().mean()
            bias = (subset[col] - subset["actual_mph"]).mean()
            records.append(
                {
                    "speed_bin": bin_label,
                    "model": model_name,
                    "MAE": mae,
                    "bias": bias,
                    "N": n,
                }
            )

    rdf = pd.DataFrame(records)
    if rdf.empty:
        log.warning("No data for speed-range plot")
        return

    fig, ax1 = plt.subplots(figsize=(12, 5))
    speed_bins_present = rdf["speed_bin"].unique()
    x = np.arange(len(speed_bins_present))
    width = 0.18
    models = list(MODEL_COLS.keys())
    colors = ["#bbb", "#6baed6", "#2ca02c", "#d62728"]

    for i, model_name in enumerate(models):
        model_data = rdf[rdf["model"] == model_name]
        vals = [
            model_data.loc[model_data["speed_bin"] == b, "MAE"].values[0]
            if b in model_data["speed_bin"].values
            else 0
            for b in speed_bins_present
        ]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax1.bar(
            x + offset, vals, width, label=model_name, color=colors[i], alpha=0.85
        )

    # Annotate N on x-axis
    ns = rdf.groupby("speed_bin")["N"].first()
    tick_labels = [f"{b}\n(n={ns.get(b, 0):,})" for b in speed_bins_present]
    ax1.set_xticks(x)
    ax1.set_xticklabels(tick_labels)
    ax1.set_ylabel("MAE (mph)")
    ax1.set_xlabel("Speed limit range (mph)")
    ax1.set_title("Mean Absolute Error by Speed Limit")
    ax1.legend(fontsize=8, ncol=2)

    # Bias line for primary model
    ax2 = ax1.twinx()
    hgb_bias = [
        rdf.loc[
            (rdf["model"] == PRIMARY_LABEL) & (rdf["speed_bin"] == b), "bias"
        ].values
        for b in speed_bins_present
    ]
    hgb_bias = [v[0] if len(v) else np.nan for v in hgb_bias]
    ax2.plot(x, hgb_bias, "ko-", markersize=5, label=f"{PRIMARY_LABEL} bias")
    ax2.axhline(0, color="gray", ls=":", lw=0.8)
    ax2.set_ylabel("Bias (mph) — predicted minus actual")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out = output_dir / "accuracy_by_speed_range.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# Plot 4 — Residuals by hour
# ---------------------------------------------------------------------------


def plot_residuals_by_hour(test_df: pd.DataFrame, output_dir: Path) -> None:
    df = test_df.dropna(subset=["hour"]).copy()
    df["hour"] = df["hour"].astype(int)
    df["residual"] = df[PRIMARY_MODEL] - df["actual_mph"]

    if df.empty:
        log.warning("No hourly data — skipping residuals-by-hour plot")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="hour", y="residual", color="steelblue", fliersize=2, ax=ax)
    ax.axhline(0, color="red", ls="--", lw=1)

    # Median line
    medians = df.groupby("hour")["residual"].median()
    hours_sorted = sorted(medians.index)
    ax.plot(
        [hours_sorted.index(h) for h in hours_sorted],
        [medians[h] for h in hours_sorted],
        "ro-",
        markersize=4,
        lw=1.5,
        label="median",
    )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Residual (predicted − actual, mph)")
    ax.set_title(f"Prediction Residuals by Hour — {PRIMARY_LABEL}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = output_dir / "residuals_by_hour.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# Plot 5 — Residuals by highway type
# ---------------------------------------------------------------------------


def plot_residuals_by_highway(test_df: pd.DataFrame, output_dir: Path) -> None:
    df = test_df.copy()
    df["abs_err"] = (df[PRIMARY_MODEL] - df["actual_mph"]).abs()

    # Order by median error descending
    order = df.groupby("highway")["abs_err"].median().sort_values(ascending=False).index

    # Count per type
    counts = df["highway"].value_counts()
    labels = [f"{h}\n(n={counts.get(h, 0)})" for h in order]

    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.7), 5))
    sns.boxplot(
        data=df,
        x="highway",
        y="abs_err",
        order=order,
        color="steelblue",
        fliersize=2,
        ax=ax,
    )
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Absolute Error (mph)")
    ax.set_xlabel("")
    ax.set_title(f"Prediction Error by Road Type — {PRIMARY_LABEL}")
    fig.tight_layout()
    out = output_dir / "residuals_by_highway.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# Plot 6 — Feature importances
# ---------------------------------------------------------------------------


def plot_feature_importances(output_dir: Path) -> None:
    fi_path = output_dir / "feature_importances.csv"
    if not fi_path.exists():
        log.warning("feature_importances.csv not found — skipping")
        return

    fi = pd.read_csv(fi_path, index_col=0).squeeze()
    fi = fi.sort_values(ascending=True).tail(15)

    # Color by category
    def _cat_color(name: str) -> str:
        if name.startswith("highway_"):
            return "#ff7f0e"
        if name in ("hour", "is_weekday", "is_peak"):
            return "#2ca02c"
        return "#1f77b4"

    colors = [_cat_color(n) for n in fi.index]

    fig, ax = plt.subplots(figsize=(8, 6))
    fi.plot.barh(ax=ax, color=colors)
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importances (top 15)")

    # Legend
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(color="#1f77b4", label="Road attribute"),
        Patch(color="#2ca02c", label="Temporal"),
        Patch(color="#ff7f0e", label="Highway type (OHE)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    out = output_dir / "feature_importances.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# Plot 7 — Residual vs feature diagnostic grid
# ---------------------------------------------------------------------------


def plot_residuals_vs_features(test_df: pd.DataFrame, output_dir: Path) -> None:
    df = test_df.copy()
    df["residual"] = df[PRIMARY_MODEL] - df["actual_mph"]

    feature_cols = ["maxspeed_mph", "lanes", "grade_abs", "link_length_km", "hour"]
    # Use scheduled_speed_mph if available, else n_stops
    if "scheduled_speed_mph" in df.columns:
        feature_cols.append("scheduled_speed_mph")
    elif "n_stops" in df.columns:
        feature_cols.append("n_stops")
    else:
        feature_cols.append("maxspeed_mph")  # duplicate as filler

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 8))
    axes = axes.ravel()

    for ax, feat in zip(axes, feature_cols):
        x = df[feat].values
        y = df["residual"].values
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 10:
            ax.set_title(feat)
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        ax.hexbin(x[valid], y[valid], gridsize=30, cmap="YlOrRd", mincnt=1)
        ax.axhline(0, color="red", ls="--", lw=1)

        # LOWESS trend
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            sorted_idx = np.argsort(x[valid])
            xs = x[valid][sorted_idx]
            ys = y[valid][sorted_idx]
            smooth = lowess(ys, xs, frac=0.3, return_sorted=True)
            ax.plot(smooth[:, 0], smooth[:, 1], "b-", lw=2, label="LOWESS")
        except ImportError:
            pass

        ax.set_xlabel(feat)
        ax.set_ylabel("Residual (mph)")
        ax.set_title(feat)

    fig.suptitle(f"Residuals vs Features — {PRIMARY_LABEL}", fontsize=13, y=1.01)
    fig.tight_layout()
    out = output_dir / "residuals_vs_features.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    data_dirs: list[Path],
    output_dir: Path | None = None,
    error_type: str = "abs",
) -> None:
    if output_dir is None:
        output_dir = data_dirs[0] if len(data_dirs) == 1 else Path("reports/realtime")

    # Load test predictions
    test_path = output_dir / "test_predictions.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"test_predictions.csv not found in {output_dir}. "
            "Run fit_speed_models.py first."
        )
    test_df = pd.read_csv(test_path)
    log.info("Loaded %d test predictions from %s", len(test_df), test_path)

    # Load train road IDs
    train_roads_path = output_dir / "train_road_ids.csv"
    train_road_ids: set[str] = set()
    if train_roads_path.exists():
        train_road_ids = set(pd.read_csv(train_roads_path)["road_id"].astype(str))
        log.info("Loaded %d training road IDs", len(train_road_ids))

    # Load all predictions (train + test) for full-network speed map
    all_path = output_dir / "all_predictions.csv"
    all_df = None
    if all_path.exists():
        all_df = pd.read_csv(all_path)
        log.info("Loaded %d all-road predictions from %s", len(all_df), all_path)
    else:
        log.warning(
            "all_predictions.csv not found — predicted-speed layer will be skipped"
        )

    multi_agency = "agency" in test_df.columns
    geom_lookup = _load_geometry_lookup(data_dirs, multi_agency=multi_agency)

    # Generate all plots
    build_error_map(
        test_df, train_road_ids, geom_lookup, output_dir, error_type, all_df=all_df
    )
    plot_predicted_vs_actual(test_df, output_dir)
    plot_accuracy_by_speed_range(test_df, output_dir)
    plot_residuals_by_hour(test_df, output_dir)
    plot_residuals_by_highway(test_df, output_dir)
    plot_feature_importances(output_dir)
    plot_residuals_vs_features(test_df, output_dir)

    log.info("All visualizations saved to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize speed model diagnostics")
    parser.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        dest="data_dirs",
        help="Agency data directory (may be repeated)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory containing model outputs (default: first data-dir or reports/realtime)",
    )
    parser.add_argument(
        "--map-error-type",
        choices=["abs", "pct"],
        default="abs",
        help="Error metric for folium map: 'abs' (mph) or 'pct' (%%)",
    )
    args = parser.parse_args()
    dirs = args.data_dirs if args.data_dirs else DEFAULT_DATA_DIRS
    main(dirs, args.output_dir, args.map_error_type)
