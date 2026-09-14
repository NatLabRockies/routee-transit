"""Energy analysis for all six bundled RouteE-Transit bus models.

Evaluates a coarse grid of "fake" road links spanning speed (x-axis) and road
grade (y-axis) for each vehicle model, using RouteE-Compass as the evaluator,
and renders three figures:

    * a 2x3 grid of heatmaps -- per-fuel metric:
        - battery-electric buses -> kWh / mile
        - diesel / CNG buses     -> MPGde (miles per diesel gallon equivalent)
    * energy consumption vs. grade (all six buses, speed held at 30 kph)
    * energy consumption vs. speed (all six buses, grade held at 0%)

The two line charts compare all six vehicles on one axis, so they use a common
metric -- kWh/mile (the raw powertrain energy, which is in kWh for every model).

No OSM data is downloaded. We borrow an existing compass_app graph cache only
for its topology, then overwrite every edge's posted-speed and grade with a
controlled grid and blank the GTFS stop mapping so no kinetic stop penalty
contaminates the pure powertrain rate. Each grid cell is one edge; we query
single-edge paths and read edge_energy / edge_distance. Edge length cancels out
of the per-mile rate, so the borrowed edge geometries don't matter.

matplotlib is not in the Compass (pixi) environment, so this runs in two stages:

    # 1) compute the grid with Compass (writes a CSV) -- needs routee.transit
    .pixi/envs/dev-py312/bin/python scripts/vehicle_model_energy_heatmaps.py --compute-only

    # 2) render the figures from the CSV -- needs matplotlib
    /Users/dmccabe/miniconda3/bin/python scripts/vehicle_model_energy_heatmaps.py --plot-only
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

VEHICLE_DIR = REPO_ROOT / "routee/transit/resources/vehicle_models"
BASE_COMPASS_APP = REPO_ROOT / "reports/saltlake/compass_app"

OUT_DIR = REPO_ROOT / "scripts/figures"
CSV_OUT = OUT_DIR / "vehicle_model_energy_grid.csv"
HEATMAP_OUT = OUT_DIR / "vehicle_model_energy_heatmaps.png"
LINES_OUT = OUT_DIR / "vehicle_model_energy_lines.png"

# EPA/DOE gasoline gallon equivalent conversion factors (mirrors predictor.py).
KWH_PER_GGE = 33.7
GGE_PER_GALLON_DIESEL = 1.136
KWH_PER_GALLON_DIESEL = GGE_PER_GALLON_DIESEL * KWH_PER_GGE  # ~38.28 kWh/gal

# The six bundled models. "electric" reads edge_energy_electric; everything else
# reads edge_energy_liquid. Both raw fields are in kWh for these custom models.
MODELS: list[tuple[str, str]] = [
    ("Transit_Bus_Electric_40ft_300kWh", "electric"),
    ("Transit_Bus_Electric_60ft_600kWh", "electric"),
    ("Transit_Bus_Diesel_40ft", "liquid"),
    ("Transit_Bus_Diesel_60ft", "liquid"),
    ("Transit_Bus_CNG_40ft", "liquid"),
    ("Transit_Bus_CNG_60ft", "liquid"),
]

# Coarse grid to show general trends. Speeds include 30 kph and grades include
# 0% so the two line charts can slice the grid directly. Both ranges stay inside
# every model's feature bounds (speed 0-160 kph, grade +/-0.2).
SPEEDS_KPH = np.array([5.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 100.0])
GRADES = np.round(np.arange(-0.08, 0.0801, 0.02), 4)  # -8%..+8% by 2%

# Anchors for the two line charts (must be grid values).
LINE_SPEED_KPH = 30.0
LINE_GRADE = 0.0

# Consistent per-vehicle styling for the line charts.
MODEL_STYLE: dict[str, dict[str, str]] = {
    "Transit_Bus_Electric_40ft_300kWh": {"color": "#1f77b4", "ls": "-"},
    "Transit_Bus_Electric_60ft_600kWh": {"color": "#17becf", "ls": "--"},
    "Transit_Bus_Diesel_40ft": {"color": "#d62728", "ls": "-"},
    "Transit_Bus_Diesel_60ft": {"color": "#ff7f0e", "ls": "--"},
    "Transit_Bus_CNG_40ft": {"color": "#2ca02c", "ls": "-"},
    "Transit_Bus_CNG_60ft": {"color": "#8c9440", "ls": "--"},
}


def _label(model_name: str) -> str:
    return model_name.replace("Transit_Bus_", "").replace("_", " ")


def _write_gz_column(path: Path, values: list[str]) -> None:
    with gzip.open(path, "wt") as f:
        f.write("\n".join(values) + "\n")


def _setup_base_scratch(scratch: Path, grid: list[tuple[float, float]]) -> None:
    """Symlink the borrowed graph and overwrite speed/grade tables + stops."""
    scratch.mkdir(parents=True, exist_ok=True)

    override = {
        "edges-posted-speed-enumerated.txt.gz",
        "edges-grade-enumerated.txt.gz",
        "gtfs_stops.csv",
        "transit_energy.toml",
        "vehicles",
    }
    for entry in BASE_COMPASS_APP.iterdir():
        if entry.name in override:
            continue
        (scratch / entry.name).symlink_to(entry)

    with gzip.open(
        BASE_COMPASS_APP / "edges-posted-speed-enumerated.txt.gz", "rt"
    ) as f:
        speeds = f.read().splitlines()
    with gzip.open(BASE_COMPASS_APP / "edges-grade-enumerated.txt.gz", "rt") as f:
        grades = f.read().splitlines()
    for edge_id, (speed, grade) in enumerate(grid):
        speeds[edge_id] = repr(float(speed))
        grades[edge_id] = repr(float(grade))
    _write_gz_column(scratch / "edges-posted-speed-enumerated.txt.gz", speeds)
    _write_gz_column(scratch / "edges-grade-enumerated.txt.gz", grades)

    # Blank GTFS stop mapping -> no kinetic stop penalty (pure powertrain rate).
    (scratch / "gtfs_stops.csv").write_text("trip_id,edge_id\n", encoding="utf-8")


def _write_model_config(scratch: Path, model_name: str) -> Path:
    """Point the scratch app at a single vehicle model and return its config."""
    import tomlkit

    vehicles_dir = scratch / "vehicles"
    if vehicles_dir.exists():
        shutil.rmtree(vehicles_dir)
    vehicles_dir.mkdir()

    vehicle_json = VEHICLE_DIR / f"{model_name}.json"
    shutil.copy(vehicle_json, vehicles_dir / vehicle_json.name)
    bin_name = json.loads(vehicle_json.read_text())["model_input_file"]
    shutil.copy(VEHICLE_DIR / bin_name, vehicles_dir / bin_name)

    config = tomlkit.loads((BASE_COMPASS_APP / "transit_energy.toml").read_text())
    for model in config["search"]["traversal"]["models"]:  # type: ignore[index]
        if model.get("type") == "transit_energy":
            model["vehicle_input_files"] = [f"vehicles/{vehicle_json.name}"]
    config_path = scratch / "transit_energy.toml"
    with open(config_path, "w") as f:
        tomlkit.dump(config, f)
    return config_path


def compute_grid() -> pd.DataFrame:
    from routee.transit.compass_app import TransitCompassApp

    grid = [(float(s), float(g)) for s in SPEEDS_KPH for g in GRADES]
    n_cells = len(grid)

    scratch = Path(tempfile.mkdtemp(prefix="vehicle_heatmap_"))
    frames: list[pd.DataFrame] = []
    try:
        _setup_base_scratch(scratch, grid)

        for model_name, fuel in MODELS:
            print(f"Evaluating {model_name} ({fuel})...")
            config_path = _write_model_config(scratch, model_name)
            app = TransitCompassApp.from_config_file(config_path, parallelism=4)

            energy_field = (
                "edge_energy_electric" if fuel == "electric" else "edge_energy_liquid"
            )
            queries = [
                {
                    "path": [{"edge_id": edge_id}],
                    "model_name": model_name,
                    "weights": {"trip_time": 1.0},
                    "start_time": "08:00:00",
                    "start_weekday": "monday",
                }
                for edge_id in range(n_cells)
            ]
            results = app.run_calculate_path(queries)
            if isinstance(results, dict):
                results = [results]

            rows = []
            for (speed, grade), result in zip(grid, results):
                energy_kwh = dist_mi = np.nan
                if "error" not in result:
                    summary = result.get("route", {}).get("traversal_summary", {})
                    energy_kwh = float(
                        summary.get(energy_field, {}).get("value", np.nan)
                    )
                    dist_mi = float(
                        summary.get("edge_distance", {}).get("value", np.nan)
                    )
                rows.append((model_name, fuel, speed, grade, energy_kwh, dist_mi))

            frames.append(
                pd.DataFrame(
                    rows,
                    columns=[
                        "model",
                        "fuel",
                        "speed_kph",
                        "grade",
                        "energy_kwh",
                        "dist_mi",
                    ],
                )
            )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    df = pd.concat(frames, ignore_index=True)
    # Derived metrics: kWh/mile (common to all), MPGde (combustion economy).
    df["kwh_per_mile"] = df["energy_kwh"] / df["dist_mi"]
    df["mpgde"] = df["dist_mi"] * KWH_PER_GALLON_DIESEL / df["energy_kwh"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"Wrote {len(df)} grid points to {CSV_OUT}")
    return df


def plot_heatmaps(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    plt.rcParams.update(
        {"font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold"}
    )

    speeds = np.sort(df["speed_kph"].unique())
    grades = np.sort(df["grade"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)

    for ax, (model_name, fuel) in zip(axes.flat, MODELS):
        sub = df[df["model"] == model_name]
        value_col = "kwh_per_mile" if fuel == "electric" else "mpgde"
        pivot = sub.pivot(index="grade", columns="speed_kph", values=value_col).reindex(
            index=grades, columns=speeds
        )
        z = pivot.to_numpy()

        if fuel == "electric":
            cmap, cbar_label = colormaps["viridis"], "kWh / mile"
        else:
            cmap, cbar_label = colormaps["magma"], "MPGde"

        mesh = ax.imshow(
            z,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=(-0.5, len(speeds) - 0.5, -0.5, len(grades) - 0.5),
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label(cbar_label, fontsize=9)

        # Annotate each coarse cell with its value.
        zmin, zmax = np.nanmin(z), np.nanmax(z)
        for i in range(len(grades)):
            for j in range(len(speeds)):
                val = z[i, j]
                if np.isnan(val):
                    continue
                norm = (val - zmin) / (zmax - zmin + 1e-9)
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if norm < 0.55 else "black",
                )

        ax.set_xticks(range(len(speeds)))
        ax.set_xticklabels([f"{int(s)}" for s in speeds])
        ax.set_yticks(range(len(grades)))
        ax.set_yticklabels([f"{g * 100:+.0f}" for g in grades])
        ax.set_title(_label(model_name))
        ax.set_xlabel("Speed (kph)")
        ax.set_ylabel("Road grade (%)")

    fig.suptitle(
        "RouteE-Transit bus models: energy consumption vs. speed and grade",
        fontsize=15,
        fontweight="bold",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(HEATMAP_OUT, dpi=150, bbox_inches="tight")
    print(f"Wrote heatmap grid to {HEATMAP_OUT}")


def plot_lines(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold"}
    )

    fig, (ax_grade, ax_speed) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: energy vs. grade at a fixed speed.
    grade_slice = df[np.isclose(df["speed_kph"], LINE_SPEED_KPH)]
    for model_name, _ in MODELS:
        sub = grade_slice[grade_slice["model"] == model_name].sort_values("grade")
        style = MODEL_STYLE[model_name]
        ax_grade.plot(
            sub["grade"] * 100,
            sub["kwh_per_mile"],
            marker="o",
            ms=4,
            color=style["color"],
            ls=style["ls"],
            label=_label(model_name),
        )
    ax_grade.set_title(f"Energy vs. grade (speed = {int(LINE_SPEED_KPH)} kph)")
    ax_grade.set_xlabel("Road grade (%)")
    ax_grade.set_ylabel("Energy consumption (kWh / mile)")
    ax_grade.axhline(0, color="0.6", lw=0.8)
    ax_grade.axvline(0, color="0.6", lw=0.8)
    ax_grade.grid(True, alpha=0.3)
    # A few grid cells are model artifacts (e.g. near-zero fuel economy on steep
    # grades) that spike far off-scale; clip the view to keep trends legible.
    vals = grade_slice["kwh_per_mile"].to_numpy()
    top = np.nanpercentile(vals, 92) * 1.25
    if np.nanmax(vals) > top:
        ax_grade.set_ylim(top=top)
        ax_grade.text(
            0.99,
            0.97,
            "y-axis clipped (off-scale model artifacts)",
            transform=ax_grade.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="0.4",
        )

    # Right: energy vs. speed at flat grade.
    speed_slice = df[np.isclose(df["grade"], LINE_GRADE)]
    for model_name, _ in MODELS:
        sub = speed_slice[speed_slice["model"] == model_name].sort_values("speed_kph")
        style = MODEL_STYLE[model_name]
        ax_speed.plot(
            sub["speed_kph"],
            sub["kwh_per_mile"],
            marker="o",
            ms=4,
            color=style["color"],
            ls=style["ls"],
            label=_label(model_name),
        )
    ax_speed.set_title(f"Energy vs. speed (grade = {int(LINE_GRADE * 100)}%)")
    ax_speed.set_xlabel("Speed (kph)")
    ax_speed.set_ylabel("Energy consumption (kWh / mile)")
    ax_speed.grid(True, alpha=0.3)

    handles, labels = ax_grade.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "RouteE-Transit bus models: energy consumption trends",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(LINES_OUT, dpi=150, bbox_inches="tight")
    print(f"Wrote line charts to {LINES_OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compute-only",
        action="store_true",
        help="Only run Compass and write the grid CSV (needs routee.transit).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only render figures from the existing CSV (needs matplotlib).",
    )
    args = parser.parse_args()

    if args.plot_only:
        df = pd.read_csv(CSV_OUT)
        plot_heatmaps(df)
        plot_lines(df)
        return

    df = compute_grid()
    if args.compute_only:
        return
    try:
        plot_heatmaps(df)
        plot_lines(df)
    except ImportError:
        print(
            "\nmatplotlib not available here. CSV written; render with:\n"
            f"  python {Path(__file__).relative_to(REPO_ROOT)} --plot-only"
        )


if __name__ == "__main__":
    main()
