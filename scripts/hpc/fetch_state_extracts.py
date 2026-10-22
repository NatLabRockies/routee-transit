"""Download the Geofabrik state extracts a feed database needs.

Which states to fetch comes from the Mobility Database ``states`` column, which
is cheap but not trustworthy on its own: it is not US-only (observed values
include Canadian provinces and, apparently from a geocoding slip, the Algerian
province of Mascara). Every name is therefore validated against the Census
state list before it becomes a download URL.

This only decides what to pre-download. At run time the state files actually
read are chosen geometrically from the corridor polygon, so a feed whose
buffered corridor crosses into a neighbouring state still gets a complete
network. Anything missed here is fetched on demand.

Usage:
    python scripts/hpc/fetch_state_extracts.py /path/to/db --out-dir /projects/teta/osm/states
"""

from __future__ import annotations

import argparse
import ast
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from routee.transit.osm_states import (
    UNSUPPORTED_SLUGS,
    download_state_pbf,
    fetch_states_gdf,
    state_pbf_path,
)


def qualifying_feed_states(db_root: Path) -> list[str]:
    """Slugs named by the feed metadata for datasets we would analyze."""
    feeds = pd.read_csv(db_root / "feeds.csv")
    datasets = pd.read_csv(db_root / "datasets.csv")
    merged = feeds.merge(
        datasets[
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

    names: set[str] = set()
    for value in qualifying["states"].dropna():
        try:
            names.update(ast.literal_eval(value))
        except (ValueError, SyntaxError):
            continue
    return sorted(name.lower().replace(" ", "-") for name in names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent downloads (default: 4). Geofabrik throttles per connection.",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    claimed = qualifying_feed_states(args.db_root)
    valid = set(fetch_states_gdf(args.out_dir / "_boundaries")["slug"])
    slugs = sorted((set(claimed) & valid) - UNSUPPORTED_SLUGS)
    rejected = sorted(set(claimed) - valid)

    print(f"feed metadata named {len(claimed)} region(s)")
    if rejected:
        print(f"  ignoring {len(rejected)} non-US/unknown: {', '.join(rejected)}")
    print(f"{len(slugs)} US state extract(s) needed")

    pending = [s for s in slugs if not state_pbf_path(s, args.out_dir).is_file()]
    print(f"{len(slugs) - len(pending)} already present, {len(pending)} to download")
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        return 0

    start = time.time()
    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_state_pbf, slug, args.out_dir): slug
            for slug in pending
        }
        for future in as_completed(futures):
            slug = futures[future]
            try:
                path = future.result()
            except Exception as error:
                failed += 1
                print(f"  FAIL {slug}: {type(error).__name__}: {error}", flush=True)
                continue
            done += 1
            size_mb = path.stat().st_size / 1e6
            elapsed = time.time() - start
            print(
                f"  [{done + failed}/{len(pending)}] {slug} "
                f"({size_mb:.0f} MB, {elapsed / 60:.1f} min elapsed)",
                flush=True,
            )

    total_gb = (
        sum(
            state_pbf_path(s, args.out_dir).stat().st_size
            for s in slugs
            if state_pbf_path(s, args.out_dir).is_file()
        )
        / 1e9
    )
    print(
        f"\n{done} downloaded, {failed} failed, {total_gb:.1f} GB total "
        f"in {(time.time() - start) / 60:.1f} min"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
