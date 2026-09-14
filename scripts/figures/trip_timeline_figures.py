"""
Two trip-level granularity figures for the FTA presentation.

Figure 1 — Fleet energy Gantt  (fleet_gantt.png)
    All blocks as rows, time-of-day on X, each trip a horizontal bar
    coloured by kWh/mile intensity.  Rows sorted by first departure.

Figure 2 — Block 132 energy timeline  (block_energy_timeline.png)
    Every trip in block 132's primary service day as a vertical bar;
    bar height = kWh, colour = trip type (service / deadhead variants).

Usage:
    conda run -n routee-transit-results python scripts/figures/trip_timeline_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = "mdb-167-manual-download"
DEMO = REPO_ROOT / "reports" / "fta_demo_071326" / DATASET
OUT_DIR = DEMO / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_CSV = DEMO / "results" / "trip_energy_predictions.csv"
DAY_CSV = DEMO / "results" / "trip_energy_by_day.csv"

DATE = "2026-07-15"
BEB = "Transit_Bus_Battery_Electric"
BLOCK_ID = "132"

# NLR brand colours
C_BLUE = "#0079C2"
C_ORANGE = "#EE9521"
C_GREEN = "#7DA544"
C_YELLOW = "#FFC423"
C_DGRAY = "#626D72"
BG = "#ffffff"

TRIP_TYPE_COLORS = {
    "service": C_BLUE,
    "mid_block_deadhead": C_YELLOW,
    "pull-out": C_ORANGE,
    "pull-in": C_GREEN,
}

TRIP_TYPE_LABELS = {
    "service": "Service",
    "mid_block_deadhead": "Mid-block deadhead",
    "pull-out": "Pull-out",
    "pull-in": "Pull-in",
}

# Global matplotlib style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.axisbelow": True,
        "xtick.bottom": False,
        "ytick.left": False,
        "xtick.color": "#888888",
        "ytick.color": "#888888",
        "grid.color": "#eeeeee",
        "grid.linewidth": 0.8,
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_hours(t: str) -> float:
    """Parse 'HH:MM:SS' (hours may exceed 24) to fractional hours."""
    h, m, s = str(t).split(":")
    return int(h) + int(m) / 60 + float(s) / 3600


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["block_id"] = df["block_id"].astype(str)
    df["start_h"] = df["start_time"].apply(_to_hours)
    df["end_h"] = df["end_time"].apply(_to_hours)
    df["dur_h"] = (df["end_h"] - df["start_h"]).clip(lower=1 / 60)
    df["intensity"] = df["energy_used"] / df["miles"].replace(0.0, np.nan)  # kWh/mile
    return df


# ===========================================================================
# Figure 1 — Fleet Gantt
# ===========================================================================
def plot_fleet_gantt() -> None:
    raw = pd.read_csv(DAY_CSV, low_memory=False)
    raw["date"] = pd.to_datetime(raw["date"])

    day = _prep(raw[(raw["vehicle"] == BEB) & (raw["date"] == DATE)])

    # Exclude known bad trips: 20FLEX pull-out departs 4 h before first service
    # (depot departure time computed incorrectly — to be fixed in a future run)
    EXCLUDE_TRIP_IDS = {"depot_to_1014020"}
    day = day[~day["trip_id"].isin(EXCLUDE_TRIP_IDS)]

    if day.empty:
        print(f"WARNING: no BEB trips found on {DATE} in {DAY_CSV.name}")
        return

    # Sort blocks by first departure
    first_dep = day.groupby("block_id")["start_h"].min().sort_values()
    block_order = first_dep.index.tolist()
    block_y = {bid: i for i, bid in enumerate(block_order)}

    # Colour scale: kWh/mile intensity, 2nd–98th percentile
    valid = day["intensity"].dropna()
    lo, hi = valid.quantile(0.02), valid.quantile(0.98)
    norm = mcolors.Normalize(vmin=lo, vmax=hi)
    cmap = plt.colormaps["YlOrRd"]

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("#f5f5f5")

    for _, row in day.iterrows():
        y = block_y.get(str(row["block_id"]))
        if y is None:
            continue
        rgba = cmap(norm(row["intensity"])) if pd.notna(row["intensity"]) else "#cccccc"
        ax.barh(
            y,
            row["dur_h"],
            left=row["start_h"],
            height=0.72,
            color=rgba,
            edgecolor="none",
        )

    ax.set_xlim(5.5, 25.0)
    ax.set_ylim(-0.8, len(block_order) - 0.2)
    ax.set_yticks(range(len(block_order)))
    ax.set_yticklabels(block_order, fontsize=8, color="#444444")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda h, _: f"{int(h % 24):02d}:00")
    )
    ax.tick_params(axis="x", labelsize=9.5)
    ax.xaxis.grid(True, zorder=0)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.015, shrink=0.55, aspect=22)
    cbar.set_label("kWh / mile", fontsize=9, color="#444444", labelpad=7)
    cbar.ax.tick_params(labelsize=8.5, colors="#555555")
    cbar.outline.set_edgecolor("#cccccc")

    plt.tight_layout(pad=1.0)
    out = OUT_DIR / "fleet_gantt.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {out}  ({len(block_order)} blocks, {len(day)} trip rows)")
    plt.close(fig)


# ===========================================================================
# Figure 2 — Block 132 energy timeline
# ===========================================================================
def plot_block_timeline() -> None:
    raw = pd.read_csv(DAY_CSV, low_memory=False)
    raw["block_id"] = raw["block_id"].astype(str)
    raw["date"] = pd.to_datetime(raw["date"])

    b = _prep(
        raw[
            (raw["block_id"] == BLOCK_ID)
            & (raw["vehicle"] == BEB)
            & (raw["date"] == DATE)
        ]
    ).sort_values("start_h")

    if b.empty:
        print(f"WARNING: no rows found for block {BLOCK_ID} on {DATE}")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for _, row in b.iterrows():
        color = TRIP_TYPE_COLORS.get(row["trip_type"], C_DGRAY)
        ax.bar(
            row["start_h"],
            row["energy_used"],
            width=row["dur_h"] * 0.85,
            align="edge",
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlim(5.5, 23.0)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda h, _: f"{int(h):02d}:00"))
    ax.tick_params(axis="x", labelsize=9.5)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel("Energy  (kWh)", fontsize=10, color="#555555", labelpad=6)
    ax.yaxis.grid(True, zorder=0)

    # Compact legend — only types that actually appear
    seen = b["trip_type"].unique()
    handles = [
        mpatches.Patch(
            facecolor=TRIP_TYPE_COLORS[t], edgecolor="none", label=TRIP_TYPE_LABELS[t]
        )
        for t in ["pull-out", "service", "mid_block_deadhead", "pull-in"]
        if t in seen
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=8.5,
        frameon=False,
        ncol=len(handles),
        borderpad=0.4,
        handlelength=1.2,
        handletextpad=0.5,
    )

    plt.tight_layout(pad=0.8)
    out = OUT_DIR / "block_energy_timeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {out}  (block {BLOCK_ID}, {len(b)} trips)")
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    plot_fleet_gantt()
    plot_block_timeline()
