"""Aggregate realtime link speeds across all days of service for a transit agency.

This script processes every JSONL file in a given realtime data directory,
computes per-link speed estimates for each day, and then produces a combined
multi-day aggregation.

It reuses the helper functions from ``aggregate_speeds.py`` and builds the
CompassApp only once for efficiency.

Usage
-----
    python aggregate_portland_me.py [--data-dir PATH]
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from aggregate_speeds import (
    aggregate_speeds_across_trips,
    build_compass_app,
    get_link_speeds_for_trip,
    read_realtime_records,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("reports/realtime/greater_portland_me")


def get_speeds_for_day(
    path_to_json: Path,
    app,
    edge_attr_df: pd.DataFrame,
    trips_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """Process a single day's JSONL file and return per-trip link speeds.

    Modelled on ``aggregate_speeds.get_speeds_for_one_day`` but accepts a
    pre-built CompassApp so it can be reused across days.
    """
    rt_df = read_realtime_records(path_to_json)
    n_trips = rt_df["trip_id"].nunique()
    log.info("  %d trips in %s", n_trips, path_to_json.name)

    if n_trips == 0:
        return None

    if "route_id" not in rt_df.columns:
        rt_df = rt_df.merge(trips_df[["route_id"]], left_on="trip_id", right_index=True)

    all_results: list[pd.DataFrame] = []
    for route_id, route_rt in rt_df.groupby("route_id"):
        trip_ids = list(route_rt["trip_id"].unique())
        log.info("    route %s – %d trips", route_id, len(trip_ids))

        results = []
        for tid in trip_ids:
            result = get_link_speeds_for_trip(
                tid,
                rt_df=rt_df,
                trips_df=trips_df,
                app=app,
                edge_attr_df=edge_attr_df,
            )
            if not result.empty and "_skip_reason" not in result.columns:
                results.append(result)

        if results:
            all_results.append(pd.concat(results, ignore_index=True))

    if not all_results:
        return None

    return pd.concat(all_results, ignore_index=True)


def main(data_dir: Path) -> None:
    gtfs_root = data_dir

    # --- Read static GTFS files -----------------------------------------------
    trips_df = pd.read_csv(
        gtfs_root / "static/trips.txt",
        dtype={"trip_id": str, "shape_id": str},
    ).set_index("trip_id")
    shapes_df = pd.read_csv(gtfs_root / "static/shapes.txt", dtype={"shape_id": str})
    log.info(
        "Loaded static GTFS: %d trips, %d shape points", len(trips_df), len(shapes_df)
    )

    # --- Build CompassApp once ------------------------------------------------
    log.info("Building CompassApp (one-time)…")
    app, edge_attr_df = build_compass_app(shapes_df)
    log.info("CompassApp ready – %d edges", len(edge_attr_df))

    # --- Discover JSONL files -------------------------------------------------
    jsonl_files = sorted(gtfs_root.glob("gtfs_realtime_records_*.jsonl"))
    log.info("Found %d JSONL files to process", len(jsonl_files))

    # --- Process each day -----------------------------------------------------
    for jf in jsonl_files:
        file_date = jf.stem.split("_")[-1]  # e.g. "20251023"
        per_day_csv = gtfs_root / f"realtime_link_speeds_{file_date}.csv"

        if per_day_csv.exists():
            log.info(
                "Day %s already processed → %s (skipping)", file_date, per_day_csv.name
            )
            continue

        log.info("Processing day %s …", file_date)
        t0 = time.time()

        day_df = get_speeds_for_day(jf, app, edge_attr_df, trips_df)
        if day_df is not None and not day_df.empty:
            day_df["date"] = file_date
            day_df.to_csv(per_day_csv, index=False)
            log.info(
                "  saved %d rows → %s  (%.1f s)",
                len(day_df),
                per_day_csv.name,
                time.time() - t0,
            )
        else:
            log.warning(
                "  no usable data for day %s  (%.1f s)", file_date, time.time() - t0
            )

    # --- Combine all per-day CSVs ---------------------------------------------
    per_day_csvs = sorted(gtfs_root.glob("realtime_link_speeds_2*.csv"))
    # Exclude already-aggregated files
    per_day_csvs = [f for f in per_day_csvs if "aggregated" not in f.name]
    log.info("Combining %d per-day CSVs", len(per_day_csvs))

    if not per_day_csvs:
        log.warning("No per-day CSVs found — nothing to aggregate.")
        return

    combined = pd.concat([pd.read_csv(f) for f in per_day_csvs], ignore_index=True)
    combined_path = gtfs_root / "realtime_link_speeds_all_days.csv"
    combined.to_csv(combined_path, index=False)
    log.info(
        "Combined per-trip speeds: %d rows → %s", len(combined), combined_path.name
    )

    # --- Aggregate across all trips -------------------------------------------
    aggregated = aggregate_speeds_across_trips(combined)
    if not aggregated.empty:
        agg_path = gtfs_root / "realtime_link_speeds_aggregated_all_days.csv"
        aggregated.to_csv(agg_path, index=False)
        log.info(
            "Aggregated link speeds: %d road segments → %s",
            len(aggregated),
            agg_path.name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate realtime speeds for a transit agency"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root directory containing JSONL files and static/ GTFS",
    )
    args = parser.parse_args()
    main(args.data_dir)
