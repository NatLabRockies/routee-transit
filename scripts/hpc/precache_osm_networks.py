"""Pre-cache OSM networks for qualifying GTFS datasets.

Downloads OpenStreetMap road networks via the Overpass API and builds each
dataset's CompassApp cache, so the parallel analysis phase never touches
Overpass. Running this first is what keeps the public Overpass server from
refusing connections mid-run.

The cache built here must match what ``analyze_full_feed.py`` asks for -- the
same vehicle models and the same transit speed model -- otherwise the analysis
phase rebuilds it and hits Overpass anyway.

Concurrency defaults to 2. The public Overpass server grants ~2 download slots
per IP, and sustained heavier use gets the node IP blocked for minutes to hours.

Usage:
    python scripts/hpc/precache_osm_networks.py /path/to/routee-transit-db \\
        [--parallel-downloads 2] [--n_processes 1]
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import osmnx as ox
import pandas as pd

from routee.transit import GTFSEnergyPredictor
from routee.transit.osm_network import is_transient_overpass_error, overpass_endpoints
from routee.transit.predictor import _speed_model_cache_is_valid
from routee.transit.speed_model import DEFAULT_SPEED_MODEL_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_full_feed import VEHICLE_MODELS  # noqa: E402


def get_qualifying_datasets(db_root: Path) -> list[str]:
    """Read feeds.csv + datasets.csv and return qualifying dataset IDs."""
    feeds_path = db_root / "feeds.csv"
    datasets_path = db_root / "datasets.csv"

    if not feeds_path.exists() or not datasets_path.exists():
        print(f"Error: feeds.csv or datasets.csv not found in {db_root}")
        sys.exit(1)

    feeds_df = pd.read_csv(feeds_path)
    datasets_df = pd.read_csv(datasets_path)

    merged = feeds_df.merge(
        datasets_df[
            [
                "id",
                "has_shapes",
                "has_errors",
                "includes_bus_trips",
                "includes_all_bus_shapes",
            ]
        ],
        left_on="latest_dataset_id",
        right_on="id",
        how="left",
    )

    qualifying = merged[
        (merged["has_shapes"] == True)  # noqa: E712
        & (merged["has_errors"] != True)  # noqa: E712
        & (merged["includes_bus_trips"] == True)  # noqa: E712
        & (merged["includes_all_bus_shapes"] == True)  # noqa: E712
    ]

    dataset_ids = []
    for dataset_id in qualifying["latest_dataset_id"].tolist():
        gtfs_dir = db_root / dataset_id / "gtfs"
        if gtfs_dir.is_dir():
            dataset_ids.append(dataset_id)

    return dataset_ids


def is_cached(db_root: Path, dataset_id: str, speed_model_dir: Path) -> bool:
    """Check whether a dataset has a cache the analysis phase will actually reuse.

    Mirrors ``GTFSEnergyPredictor.load_compass_app``: a config that predates the
    requested vehicle models is refreshed in place without re-downloading OSM,
    but a missing or stale speed model forces a full rebuild, and therefore a
    fresh Overpass download.
    """
    cache_dir = db_root / dataset_id / "results" / "compass_app"
    if not (cache_dir / "transit_energy.toml").exists():
        return False
    return _speed_model_cache_is_valid(cache_dir, speed_model_dir)


def precache_network(
    db_root: Path,
    dataset_id: str,
    speed_model_dir: Path,
    n_processes: int,
    max_attempts: int,
    retry_wait: float,
) -> bool:
    """Download and cache the OSM network for a single dataset.

    Returns True on success, False on failure.
    """
    gtfs_dir = db_root / dataset_id / "gtfs"
    out_dir = db_root / dataset_id / "results"

    for attempt in range(1, max_attempts + 1):
        try:
            predictor = GTFSEnergyPredictor(
                gtfs_path=gtfs_dir,
                n_processes=n_processes,
                vehicle_models=VEHICLE_MODELS,
                output_dir=out_dir,
                dataset_id=dataset_id,
                speed_model_dir=speed_model_dir,
                overwrite=False,
            )
            predictor.load_gtfs_data()
            predictor.load_compass_app()
            return True
        except Exception as error:
            if attempt == max_attempts or not is_transient_overpass_error(error):
                print(f"  [{dataset_id}] ERROR: {error}", flush=True)
                return False
            wait = retry_wait * attempt
            print(
                f"  [{dataset_id}] transient failure "
                f"(attempt {attempt}/{max_attempts}): {error} \u2014 "
                f"retrying in {wait:.0f}s",
                flush=True,
            )
            time.sleep(wait)
    return False


def _precache_worker(
    db_root: Path,
    dataset_id: str,
    speed_model_dir: Path,
    n_processes: int,
    max_attempts: int,
    retry_wait: float,
    cache_folder: str,
    requests_timeout: int,
) -> tuple[str, bool]:
    """Process-pool entry point; osmnx settings don't carry over to spawned workers."""
    ox.settings.cache_folder = cache_folder
    ox.settings.requests_timeout = requests_timeout
    success = precache_network(
        db_root, dataset_id, speed_model_dir, n_processes, max_attempts, retry_wait
    )
    return dataset_id, success


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-cache OSM networks for GTFS datasets so the analysis phase "
            "never calls Overpass."
        )
    )
    parser.add_argument("db_root", type=Path, help="Path to routee-transit-db root")
    parser.add_argument(
        "--parallel-downloads",
        type=int,
        default=2,
        help=(
            "Number of datasets to process concurrently, each in its own "
            "process (default: 2). With Overpass, keep this at ~2: the public "
            "server grants ~2 slots per IP and heavier use gets the IP blocked. "
            "With local PBF extracts ($ROUTEE_OSM_PBF_DIR) it is bounded only "
            "by CPU and memory."
        ),
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=1,
        help="Processes for compass app building per dataset (default: 1)",
    )
    parser.add_argument(
        "--speed_model_dir",
        type=Path,
        default=None,
        help="Speed model bundle directory (default: the bundled baseline)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Attempts per dataset before giving up (default: 5)",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=60.0,
        help="Base seconds to wait before a retry; scales with attempt number",
    )
    parser.add_argument(
        "--pause-between",
        type=float,
        default=0.0,
        help="Seconds to pause between dataset submissions, to pace Overpass",
    )
    parser.add_argument(
        "--osmnx-cache",
        type=Path,
        default=None,
        help=(
            "Directory for osmnx's Overpass response cache. Defaults to "
            "$ROUTEE_OSMNX_CACHE, else osmnx's own default (./cache)."
        ),
    )
    parser.add_argument(
        "--requests-timeout",
        type=int,
        default=300,
        help=(
            "Per-request Overpass timeout in seconds (default: 300). Must be an "
            "integer -- osmnx interpolates it into the query as [timeout:N], and "
            "Overpass rejects a non-integer with 400 Bad Request."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional CSV path for per-dataset pre-cache outcomes",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Report cache coverage without downloading. Exits non-zero if any "
            "qualifying dataset lacks a reusable cache."
        ),
    )
    args = parser.parse_args()

    cache_folder = args.osmnx_cache or os.environ.get("ROUTEE_OSMNX_CACHE")
    if cache_folder:
        Path(cache_folder).mkdir(parents=True, exist_ok=True)
        ox.settings.cache_folder = str(cache_folder)
    # The community mirrors answer far slower than the main instance.
    ox.settings.requests_timeout = int(args.requests_timeout)
    print(f"osmnx cache folder: {ox.settings.cache_folder}")
    print(f"Overpass endpoints: {', '.join(overpass_endpoints())}")

    speed_model_dir = args.speed_model_dir or DEFAULT_SPEED_MODEL_DIR
    print(f"Speed model directory: {speed_model_dir}")
    print(f"Vehicle models: {', '.join(VEHICLE_MODELS)}")

    db_root = args.db_root
    dataset_ids = get_qualifying_datasets(db_root)
    print(f"Found {len(dataset_ids)} qualifying datasets")

    todo: list[str] = []
    already_cached: list[str] = []
    for dataset_id in dataset_ids:
        if is_cached(db_root, dataset_id, speed_model_dir):
            already_cached.append(dataset_id)
        else:
            todo.append(dataset_id)
    print(f"{len(already_cached)} already cached, {len(todo)} to download")

    if args.check_only:
        coverage = len(already_cached) / len(dataset_ids) if dataset_ids else 0.0
        print(f"Cache coverage: {coverage:.1%}")
        for dataset_id in todo[:20]:
            print(f"  missing: {dataset_id}")
        if len(todo) > 20:
            print(f"  ... and {len(todo) - 20} more")
        sys.exit(1 if todo else 0)

    outcomes: dict[str, str] = {did: "cached" for did in already_cached}
    downloaded = 0
    failed = 0

    if todo:
        n_workers = max(1, min(args.parallel_downloads, len(todo)))
        print(
            f"Downloading {len(todo)} dataset(s) with {n_workers} parallel "
            f"worker(s); {args.n_processes} process(es) per dataset."
        )

        done_count = 0
        pending = list(todo)
        while pending:
            crashed: list[str] = []
            # Separate processes rather than threads: building a network from a
            # local PBF is CPU-bound Python, so threads serialize on the GIL.
            # Each worker handles one dataset and then exits to free its memory.
            with ProcessPoolExecutor(
                max_workers=n_workers, max_tasks_per_child=1
            ) as pool:
                future_ids = {
                    pool.submit(
                        _precache_worker,
                        db_root,
                        dataset_id,
                        speed_model_dir,
                        args.n_processes,
                        args.max_attempts,
                        args.retry_wait,
                        str(ox.settings.cache_folder),
                        ox.settings.requests_timeout,
                    ): dataset_id
                    for dataset_id in pending
                }
                for fut in as_completed(future_ids):
                    dataset_id = future_ids[fut]
                    try:
                        success = fut.result()[1]
                    except BrokenProcessPool:
                        # A worker died (e.g. OOM-killed), which fails every
                        # queued dataset too; retry those with fewer workers.
                        crashed.append(dataset_id)
                        continue
                    outcomes[dataset_id] = "downloaded" if success else "failed"
                    done_count += 1
                    tag = "OK" if success else "FAIL"
                    print(
                        f"[{done_count}/{len(todo)}] {dataset_id} — {tag}", flush=True
                    )
                    if success:
                        downloaded += 1
                    else:
                        failed += 1

            if crashed and n_workers > 1:
                n_workers = max(1, n_workers // 2)
                print(
                    f"Worker pool broke; retrying {len(crashed)} dataset(s) with "
                    f"{n_workers} worker(s).",
                    flush=True,
                )
                pending = crashed
            else:
                for dataset_id in crashed:
                    outcomes[dataset_id] = "failed"
                    failed += 1
                    done_count += 1
                    print(
                        f"[{done_count}/{len(todo)}] {dataset_id} — FAIL (crashed)",
                        flush=True,
                    )
                pending = []

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"dataset_id": did, "outcome": outcomes.get(did, "not_attempted")}
                for did in dataset_ids
            ]
        ).to_csv(args.report, index=False)
        print(f"Wrote pre-cache report to {args.report}")

    print(
        f"\nSummary: {len(already_cached)} cached, {downloaded} downloaded, "
        f"{failed} failed"
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
