"""CLI entry point for aggregating GTFS-Realtime link speeds for a transit agency.

Two modes are supported:

**Single-day** — process one JSONL file and write per-link speed outputs next to it::

    python aggregate_agency_records.py --single-day PATH/TO/gtfs_realtime_records_YYYYMMDD.jsonl

**Multi-day** — process every JSONL file in a directory, skipping already-processed
days, then combine and aggregate across all days::

    python aggregate_agency_records.py --data-dir PATH/TO/agency_dir

In both modes the directory containing the JSONL file(s) must also contain a
``static/`` subdirectory with at minimum ``trips.txt`` and ``shapes.txt``
(``stop_times.txt`` and ``stops.txt`` are optional but enable scheduled-speed
and stop-count features per link).
"""

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
from realtime_speeds import (
    aggregate_speeds_across_trips,
    build_compass_app,
    clean_trip_df,
    get_link_speeds_for_trip,
    read_realtime_records,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_static_gtfs(
    gtfs_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Load static GTFS files from *gtfs_root*/static/.

    Returns (trips_df, shapes_df, stop_times_df, stops_df).
    stop_times_df and stops_df are None when the files are absent.
    """
    trips_df = pd.read_csv(
        gtfs_root / "static/trips.txt",
        dtype={"trip_id": str, "shape_id": str},
    ).set_index("trip_id")
    shapes_df = pd.read_csv(
        gtfs_root / "static/shapes.txt", dtype={"shape_id": str}
    )
    log.info(
        "Loaded static GTFS: %d trips, %d shape points", len(trips_df), len(shapes_df)
    )

    stop_times_df: pd.DataFrame | None = None
    stops_df: pd.DataFrame | None = None
    try:
        stop_times_df = pd.read_csv(
            gtfs_root / "static/stop_times.txt",
            dtype={"trip_id": str, "stop_id": str},
        )
        stops_df = pd.read_csv(
            gtfs_root / "static/stops.txt",
            dtype={"stop_id": str},
        ).set_index("stop_id")
        log.info(
            "Loaded stop data: %d stop_times rows, %d stops",
            len(stop_times_df),
            len(stops_df),
        )
    except FileNotFoundError:
        log.warning(
            "stop_times.txt or stops.txt not found — GTFS stop features will be NaN"
        )

    return trips_df, shapes_df, stop_times_df, stops_df


# ---------------------------------------------------------------------------
# Single-day runner
# ---------------------------------------------------------------------------


def run_single_day(path_to_json: Path | os.PathLike) -> None:
    """Process a single day's JSONL file and write per-link speed CSVs.

    Reads GTFS static files from ``path_to_json.parent/static/``, builds a
    CompassApp from the shapes bounding box, processes every trip, writes
    per-route CSVs, then combines them into a single dated file and an
    aggregated file.
    """
    if not isinstance(path_to_json, Path):
        path_to_json = Path(path_to_json)

    gtfs_root = path_to_json.parent

    try:
        trips_df, shapes_df, stop_times_df, stops_df = _load_static_gtfs(gtfs_root)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"GTFS static files not found in expected location: {gtfs_root}/static. "
            "Please ensure 'trips.txt' and 'shapes.txt' exist."
        ) from e

    rt_df = read_realtime_records(path_to_json)
    log.info("%d trips on this day", rt_df["trip_id"].nunique())

    if "route_id" not in rt_df.columns:
        rt_df = rt_df.merge(trips_df[["route_id"]], left_on="trip_id", right_index=True)

    app, edge_attr_df = build_compass_app(shapes_df)

    t0 = time.time()
    all_results = []

    for ix, (route_id, route_rt) in enumerate(rt_df.groupby("route_id")):
        trip_ids = list(route_rt["trip_id"].unique())
        log.info("Analyzing %d trips on route %s", len(trip_ids), route_id)
        route_start = time.time()

        raw_results = [
            get_link_speeds_for_trip(
                tid,
                rt_df=rt_df,
                trips_df=trips_df,
                app=app,
                edge_attr_df=edge_attr_df,
                stop_times_df=stop_times_df,
                stops_df=stops_df,
            )
            for tid in trip_ids
        ]
        n_skipped = sum(
            1 for r in raw_results if r.empty or "_skip_reason" in r.columns
        )
        results = [
            r for r in raw_results if not r.empty and "_skip_reason" not in r.columns
        ]

        if n_skipped:
            sample_tid = trip_ids[0]
            sample_raw = rt_df[rt_df["trip_id"] == sample_tid]
            sample_clean = clean_trip_df(sample_raw.copy())
            extra = ""
            if not sample_raw.empty:
                null_count = (
                    sample_raw[["latitude", "longitude"]].isna().any(axis=1).sum()
                )
                extra = f" (need ≥10; lat/lon null: {null_count})"
            log.warning(
                "%d/%d trips skipped on route %s. "
                "Sample trip '%s': %d raw rows → %d after cleaning%s",
                n_skipped,
                len(trip_ids),
                route_id,
                sample_tid,
                len(sample_raw),
                len(sample_clean),
                extra,
            )

        if results:
            route_df = pd.concat(results, ignore_index=True)
            route_df.to_csv(gtfs_root / f"realtime_speeds_{route_id}.csv")
            all_results.append(route_df)

        log.info(
            "Route %s: %d trips took %.2f s",
            route_id,
            route_rt["trip_id"].nunique(),
            time.time() - route_start,
        )
        log.info(
            "Finished %d of %d routes", ix + 1, rt_df["route_id"].nunique()
        )

    file_date = path_to_json.stem.split("_")[-1]
    all_csvs = list(gtfs_root.glob("realtime_speeds_*.csv"))
    if all_csvs:
        combined_df = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
        all_speeds_file = gtfs_root / f"realtime_link_speeds_{file_date}.csv"
        combined_df.to_csv(all_speeds_file, index=False)
        for f in all_csvs:
            os.remove(f)
        log.info(
            "Combined all route CSVs into %s and deleted originals.",
            all_speeds_file.name,
        )

    if all_results:
        all_trips_df = pd.concat(all_results, ignore_index=True)
        aggregated = aggregate_speeds_across_trips(all_trips_df)
        if not aggregated.empty:
            agg_file = gtfs_root / f"realtime_link_speeds_aggregated_{file_date}.csv"
            aggregated.to_csv(agg_file, index=False)
            log.info("Aggregated link speeds saved to %s", agg_file.name)

    log.info(
        "Analyzed all %d trips in %.2f s",
        rt_df["trip_id"].nunique(),
        time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Multi-day runner
# ---------------------------------------------------------------------------


def run_multi_day(data_dir: Path) -> None:
    """Process all JSONL files in *data_dir*, skipping already-processed days.

    Builds the CompassApp once from the static GTFS shapes, iterates over
    ``gtfs_realtime_records_*.jsonl`` files (one per day), writes a per-day
    CSV for each, combines them into a single all-days CSV, and produces a
    final aggregated CSV.
    """
    gtfs_root = data_dir
    trips_df, shapes_df, stop_times_df, stops_df = _load_static_gtfs(gtfs_root)

    log.info("Building CompassApp (one-time)…")
    app, edge_attr_df = build_compass_app(shapes_df)
    log.info("CompassApp ready – %d edges", len(edge_attr_df))

    jsonl_files = sorted(gtfs_root.glob("gtfs_realtime_records_*.jsonl"))
    log.info("Found %d JSONL files to process", len(jsonl_files))

    for jf in jsonl_files:
        file_date = jf.stem.split("_")[-1]
        per_day_csv = gtfs_root / f"realtime_link_speeds_{file_date}.csv"

        if per_day_csv.exists():
            log.info(
                "Day %s already processed → %s (skipping)", file_date, per_day_csv.name
            )
            continue

        log.info("Processing day %s …", file_date)
        t0 = time.time()

        rt_df = read_realtime_records(jf)
        n_trips = rt_df["trip_id"].nunique()
        log.info("  %d trips in %s", n_trips, jf.name)

        if n_trips == 0:
            log.warning("  no trips in %s — skipping", jf.name)
            continue

        if "route_id" not in rt_df.columns:
            rt_df = rt_df.merge(
                trips_df[["route_id"]], left_on="trip_id", right_index=True
            )

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
                    stop_times_df=stop_times_df,
                    stops_df=stops_df,
                )
                if not result.empty and "_skip_reason" not in result.columns:
                    results.append(result)

            if results:
                all_results.append(pd.concat(results, ignore_index=True))

        if all_results:
            day_df = pd.concat(all_results, ignore_index=True)
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

    # Combine all per-day CSVs and aggregate
    per_day_csvs = sorted(
        f
        for f in gtfs_root.glob("realtime_link_speeds_2*.csv")
        if "aggregated" not in f.name
    )
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

    aggregated = aggregate_speeds_across_trips(combined)
    if not aggregated.empty:
        agg_path = gtfs_root / "realtime_link_speeds_aggregated_all_days.csv"
        aggregated.to_csv(agg_path, index=False)
        log.info(
            "Aggregated link speeds: %d road segments → %s",
            len(aggregated),
            agg_path.name,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate GTFS-Realtime link speeds for a transit agency."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--data-dir",
        type=Path,
        metavar="DIR",
        help=(
            "Root directory containing JSONL files and static/ GTFS "
            "(multi-day mode; skips already-processed days)"
        ),
    )
    mode.add_argument(
        "--single-day",
        type=Path,
        metavar="JSONL_FILE",
        help=(
            "Path to a single JSONL file (single-day mode; "
            "static/ must be in the same parent directory)"
        ),
    )
    args = parser.parse_args()

    if args.single_day:
        run_single_day(args.single_day)
    else:
        run_multi_day(args.data_dir)
