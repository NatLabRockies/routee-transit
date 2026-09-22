"""Join Mobility Database feed and dataset metadata for batch analysis."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: python scripts/hpc/parse_datasets_csv.py <feeds_csv> <datasets_csv>",
            file=sys.stderr,
        )
        return 1

    feeds_path = Path(sys.argv[1])
    datasets_path = Path(sys.argv[2])

    dataset_info: dict[str, dict[str, str]] = {}
    with datasets_path.open(newline="") as datasets_file:
        for row in csv.DictReader(datasets_file):
            dataset_id = row.get("id", "")
            if dataset_id:
                dataset_info[dataset_id] = {
                    "has_shapes": row.get("has_shapes", ""),
                    "has_errors": row.get("has_errors", ""),
                    "includes_bus_trips": row.get("includes_bus_trips", ""),
                    "includes_all_bus_shapes": row.get("includes_all_bus_shapes", ""),
                }

    with feeds_path.open(newline="") as feeds_file:
        for row in csv.DictReader(feeds_file):
            feed_id = row.get("id", "")
            dataset_id = row.get("latest_dataset_id", "")
            if not dataset_id:
                continue

            info = dataset_info.get(dataset_id, {})
            print(
                "\t".join(
                    [
                        feed_id,
                        dataset_id,
                        info.get("has_shapes", ""),
                        info.get("has_errors", ""),
                        info.get("includes_bus_trips", ""),
                        info.get("includes_all_bus_shapes", ""),
                    ]
                )
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
