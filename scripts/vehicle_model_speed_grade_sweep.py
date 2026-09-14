"""Sweep a transit vehicle model's energy rate across speed and grade.

Evaluates kWh/mile as a function of edge speed (kph) and road grade (decimal)
for a single RouteE-Transit vehicle model, using RouteE-Compass as the
evaluator. To get a clean grid of "fake" road links, we reuse an existing
compass_app graph cache but overwrite the per-edge posted-speed and grade
tables with a controlled grid, and blank out the GTFS stop mapping so no
kinetic stop penalty contaminates the pure powertrain rate. Each grid cell is
one edge; we query single-edge paths and read edge_energy / edge_distance.

Because matplotlib is not part of the pixi environment (but Compass is), this
runs in two stages:

    # 1) compute the grid with Compass (writes a CSV)
    pixi run -e dev-py312 python scripts/vehicle_model_speed_grade_sweep.py

    # 2) plot the CSV with a Python that has matplotlib (e.g. base conda)
    python scripts/vehicle_model_speed_grade_sweep.py --plot-only
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

VEHICLE_JSON = (
    REPO_ROOT / "routee/transit/resources/vehicle_models/Transit_Bus_Electric_60ft.json"
)
BASE_COMPASS_APP = REPO_ROOT / "reports/saltlake/compass_app"

CSV_OUT = REPO_ROOT / "scripts/figures/Transit_Bus_Electric_60ft_speed_grade.csv"
PNG_OUT = REPO_ROOT / "scripts/figures/Transit_Bus_Electric_60ft_speed_grade.png"

ENERGY_FIELD = "trip_energy_electric"  # BEV summary field, in kWh

# Grid (kept inside the model's feature bounds: speed 0-160 kph, grade +/-0.2)
SPEEDS_KPH = np.arange(2, 121, 2)  # 2..120 kph
GRADES = np.round(np.arange(-0.15, 0.1501, 0.01), 4)  # -15%..+15%

# Which slices to draw as lines
GRADE_LINES = [-0.10, -0.05, 0.0, 0.05, 0.10]
SPEED_LINES = [20, 40, 60, 80, 100]


def _write_gz_column(path: Path, values: list[str]) -> None:
    with gzip.open(path, "wt") as f:
        f.write("\n".join(values) + "\n")


def _setup_scratch_app(scratch: Path, grid: list[tuple[float, float]]) -> Path:
    """Symlink the base graph, override speed/grade tables and vehicle config."""
    import tomlkit

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

    # Load the original per-edge speed/grade columns, override grid cells.
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

    # Blank GTFS stop mapping -> no kinetic stop penalty.
    (scratch / "gtfs_stops.csv").write_text("trip_id,edge_id\n", encoding="utf-8")

    # Vehicle files + a toml referencing only this model.
    vehicles_dir = scratch / "vehicles"
    vehicles_dir.mkdir()
    shutil.copy(VEHICLE_JSON, vehicles_dir / VEHICLE_JSON.name)
    bin_name = json.loads(VEHICLE_JSON.read_text())["model_input_file"]
    shutil.copy(VEHICLE_JSON.parent / bin_name, vehicles_dir / bin_name)

    config = tomlkit.loads((BASE_COMPASS_APP / "transit_energy.toml").read_text())
    for model in config["search"]["traversal"]["models"]:  # type: ignore[index]
        if model.get("type") == "transit_energy":
            model["vehicle_input_files"] = [f"vehicles/{VEHICLE_JSON.name}"]
    config_path = scratch / "transit_energy.toml"
    with open(config_path, "w") as f:
        tomlkit.dump(config, f)
    return config_path


def compute_sweep() -> pd.DataFrame:
    from routee.transit.compass_app import TransitCompassApp

    model_name = json.loads(VEHICLE_JSON.read_text())["name"]
    grid = [(float(s), float(g)) for s in SPEEDS_KPH for g in GRADES]

    scratch = Path(tempfile.mkdtemp(prefix="vehicle_sweep_"))
    try:
        config_path = _setup_scratch_app(scratch, grid)
        app = TransitCompassApp.from_config_file(config_path, parallelism=4)

        queries = [
            {
                "path": [{"edge_id": edge_id}],
                "model_name": model_name,
                "weights": {"trip_time": 1.0},
                "start_time": "08:00:00",
                "start_weekday": "monday",
            }
            for edge_id in range(len(grid))
        ]
        results = app.run_calculate_path(queries)
        if isinstance(results, dict):
            results = [results]

        rows = []
        for (speed, grade), result in zip(grid, results):
            if "error" in result:
                rows.append((speed, grade, np.nan, np.nan, np.nan))
                continue
            summary = result.get("route", {}).get("traversal_summary", {})
            energy = float(summary.get("edge_energy_electric", {}).get("value", np.nan))
            dist = float(summary.get("edge_distance", {}).get("value", np.nan))
            rate = energy / dist if dist else np.nan
            rows.append((speed, grade, rate, energy, dist))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    df = pd.DataFrame(
        rows,
        columns=[
            "speed_kph",
            "grade",
            "rate_kwh_per_mile",
            "energy_kwh",
            "distance_mi",
        ],
    )
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"Wrote {len(df)} grid points to {CSV_OUT}")

    # Quick sanity table
    flat = df[df["grade"] == 0.0].set_index("speed_kph")["rate_kwh_per_mile"]
    print("\nkWh/mile at 0% grade:")
    for s in SPEED_LINES:
        if s in flat.index:
            print(f"  {s:3d} kph: {flat.loc[s]:.3f}")
    return df


def plot(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    model_name = json.loads(VEHICLE_JSON.read_text())["name"]

    # Left: rate vs speed, one line per grade level
    for g in GRADE_LINES:
        sub = df[np.isclose(df["grade"], g)].sort_values("speed_kph")
        ax1.plot(
            sub["speed_kph"],
            sub["rate_kwh_per_mile"],
            marker="",
            label=f"{g * 100:+.0f}%",
        )
    ax1.set_xlabel("Speed (kph)")
    ax1.set_ylabel("Energy rate (kWh/mile)")
    ax1.set_title("Energy rate vs. speed")
    ax1.legend(title="Grade", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: rate vs grade, one line per speed level
    for s in SPEED_LINES:
        sub = df[np.isclose(df["speed_kph"], s)].sort_values("grade")
        ax2.plot(
            sub["grade"] * 100, sub["rate_kwh_per_mile"], marker="", label=f"{s} kph"
        )
    ax2.set_xlabel("Grade (%)")
    ax2.set_ylabel("Energy rate (kWh/mile)")
    ax2.set_title("Energy rate vs. grade")
    ax2.legend(title="Speed", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(model_name)
    fig.tight_layout()
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUT, dpi=150)
    print(f"Wrote chart to {PNG_OUT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip Compass; plot from the existing CSV (needs matplotlib).",
    )
    args = parser.parse_args()

    if args.plot_only:
        df = pd.read_csv(CSV_OUT)
    else:
        df = compute_sweep()

    try:
        plot(df)
    except ImportError:
        print(
            "\nmatplotlib not available in this environment. CSV written; "
            "run the plot step with a Python that has matplotlib:\n"
            f"  python {Path(__file__).relative_to(REPO_ROOT)} --plot-only"
        )


if __name__ == "__main__":
    main()
