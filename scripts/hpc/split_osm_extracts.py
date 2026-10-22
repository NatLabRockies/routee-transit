"""Cut per-dataset OSM extracts from a national PBF in a single pass.

A ``.osm.pbf`` has no spatial index, so a bbox read costs a full scan of the
file -- and pyrosm's cost grows superlinearly with file size (0.1s for a 44 MB
state extract vs >10 min for the 12 GB US file). Reading the national file once
per dataset is therefore not viable.

``osmium extract`` solves this: given a config listing many output regions it
cuts all of them while reading the source file once. Each dataset then reads its
own small extract, back in the fast regime.

Run this once before pre-caching; the extracts are reusable until the GTFS feeds
or the source PBF change.

Usage:
    python scripts/hpc/split_osm_extracts.py /path/to/db /projects/teta/osm/us-latest.osm.pbf \\
        --out-dir /projects/teta/osm/extracts
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd

from routee.transit import GTFSEnergyPredictor
from routee.transit.gtfs_processing import build_corridor_polygon

# osmium holds per-extract buffers in memory, so very large configs are split
# into several passes rather than one enormous one.
EXTRACTS_PER_PASS = 150


def get_qualifying_datasets(db_root: Path) -> list[str]:
    """Dataset IDs with shapes, no errors, and complete bus-trip coverage."""
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
    return [
        dataset_id
        for dataset_id in qualifying["latest_dataset_id"].tolist()
        if (db_root / dataset_id / "gtfs").is_dir()
    ]


def write_corridor_geojson(
    db_root: Path, dataset_id: str, dest: Path, buffer_deg: float, deadhead_deg: float
) -> bool:
    """Write one dataset's corridor polygon as GeoJSON. False if unavailable."""
    try:
        predictor = GTFSEnergyPredictor(
            gtfs_path=db_root / dataset_id / "gtfs", n_processes=1, output_dir=None
        )
        predictor.load_gtfs_data()
        polygon = build_corridor_polygon(
            predictor.shapes, None, buffer_deg, deadhead_deg
        )
    except Exception as error:
        print(f"  {dataset_id}: skipped ({type(error).__name__}: {error})", flush=True)
        return False

    gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326").to_file(
        dest, driver="GeoJSON"
    )
    return True


def run_osmium(config_path: Path, source_pbf: Path, strategy: str) -> None:
    subprocess.run(
        [
            "osmium",
            "extract",
            "--config",
            str(config_path),
            "--strategy",
            strategy,
            "--overwrite",
            str(source_pbf),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_root", type=Path)
    parser.add_argument("source_pbf", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--buffer-deg", type=float, default=0.01)
    parser.add_argument("--deadhead-buffer-deg", type=float, default=0.05)
    parser.add_argument(
        "--strategy",
        default="complete_ways",
        choices=["simple", "complete_ways", "smart"],
        help=(
            "osmium extract strategy. complete_ways keeps ways whole across the "
            "boundary, which routing needs; simple is faster but clips them."
        ),
    )
    parser.add_argument("--extracts-per-pass", type=int, default=EXTRACTS_PER_PASS)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if shutil.which("osmium") is None:
        print(
            "Error: osmium not on PATH. Install with: pixi global install osmium-tool"
        )
        return 1
    if not args.source_pbf.is_file():
        print(f"Error: source PBF not found: {args.source_pbf}")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    poly_dir = args.out_dir / "_polygons"
    poly_dir.mkdir(exist_ok=True)

    dataset_ids = get_qualifying_datasets(args.db_root)
    if args.limit:
        dataset_ids = dataset_ids[: args.limit]
    print(f"{len(dataset_ids)} qualifying dataset(s)")

    pending = [d for d in dataset_ids if not (args.out_dir / f"{d}.osm.pbf").is_file()]
    print(f"{len(dataset_ids) - len(pending)} already extracted, {len(pending)} to cut")
    if not pending:
        return 0

    print("Building corridor polygons...")
    regions: list[dict[str, object]] = []
    for dataset_id in pending:
        geojson = poly_dir / f"{dataset_id}.geojson"
        if not geojson.is_file() and not write_corridor_geojson(
            args.db_root, dataset_id, geojson, args.buffer_deg, args.deadhead_buffer_deg
        ):
            continue
        regions.append(
            {
                "output": f"{dataset_id}.osm.pbf",
                "polygon": {"file_name": str(geojson), "file_type": "geojson"},
            }
        )
    print(f"{len(regions)} polygon(s) ready")

    passes = [
        regions[i : i + args.extracts_per_pass]
        for i in range(0, len(regions), args.extracts_per_pass)
    ]
    for index, batch in enumerate(passes, start=1):
        config = {"directory": str(args.out_dir), "extracts": batch}
        config_path = args.out_dir / f"_extract_config_{index}.json"
        config_path.write_text(json.dumps(config, indent=2))
        print(f"\npass {index}/{len(passes)}: {len(batch)} extracts", flush=True)
        run_osmium(config_path, args.source_pbf, args.strategy)

    produced = sorted(args.out_dir.glob("*.osm.pbf"))
    total_gb = sum(p.stat().st_size for p in produced) / 1e9
    print(f"\n{len(produced)} extract(s) in {args.out_dir} ({total_gb:.1f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
