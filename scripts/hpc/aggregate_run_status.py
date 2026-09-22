"""Aggregate per-dataset status.json files into a single run_status.csv.

Scans all dataset directories in the database root, reads status.json where
present, and writes a consolidated CSV. Datasets with no status.json are
marked as "not_attempted".

Usage:
    python scripts/hpc/aggregate_run_status.py /path/to/routee-transit-db \
        [--run_id ID] [--started_at TS]
"""

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def find_dataset_dirs(db_root: Path) -> list[Path]:
    """Find all dataset directories (contain a 'gtfs' subdirectory)."""
    dirs = []
    for entry in sorted(os.listdir(db_root)):
        entry_path = db_root / entry
        if entry_path.is_dir() and (entry_path / "gtfs").is_dir():
            dirs.append(entry_path)
    return dirs


def read_status(dataset_dir: Path) -> dict[str, str | int | float | None]:
    """Read status.json from a dataset's results directory."""
    status_file = dataset_dir / "results" / "status.json"
    if status_file.exists():
        with open(status_file) as f:
            return json.load(f)
    return {}


def aggregate(db_root: Path) -> list[dict[str, str | int | float | None]]:
    """Scan all dataset dirs and collect status information."""
    rows = []
    for dataset_dir in find_dataset_dirs(db_root):
        dataset_id = dataset_dir.name
        status_data = read_status(dataset_dir)

        if status_data:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "status": status_data.get("status", "unknown"),
                    "run_id": status_data.get("run_id"),
                    "exit_code": status_data.get("exit_code"),
                    "failure_type": status_data.get("failure_type"),
                    "exception_type": status_data.get("exception_type"),
                    "error_summary": status_data.get("error_summary"),
                    "started_at": status_data.get("started_at"),
                    "completed_at": status_data.get("completed_at"),
                    "duration_minutes": status_data.get("duration_minutes"),
                }
            )
        else:
            # Has gtfs/ but no status.json — never attempted or legacy run
            has_results = (dataset_dir / "results").is_dir()
            if has_results:
                # Legacy: results dir exists but no status.json (pre-tracking run)
                has_predictions = (
                    dataset_dir / "results" / "trip_energy_predictions.csv"
                ).exists()
                status = "success (legacy)" if has_predictions else "failed (legacy)"
            else:
                status = "not_attempted"

            rows.append(
                {
                    "dataset_id": dataset_id,
                    "status": status,
                    "run_id": None,
                    "exit_code": None,
                    "failure_type": None,
                    "exception_type": None,
                    "error_summary": None,
                    "started_at": None,
                    "completed_at": None,
                    "duration_minutes": None,
                }
            )

    return rows


def write_run_status_csv(
    rows: list[dict[str, str | int | float | None]], db_root: Path
) -> None:
    """Write the aggregated run_status.csv."""
    output_path = db_root / "run_status.csv"
    fieldnames = [
        "dataset_id",
        "status",
        "run_id",
        "exit_code",
        "failure_type",
        "exception_type",
        "error_summary",
        "started_at",
        "completed_at",
        "duration_minutes",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    from collections import Counter

    counts = Counter(row["status"] for row in rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")


def append_runs_csv(
    db_root: Path,
    run_id: str | None,
    started_at: str | None,
    rows: list[dict[str, str | int | float | None]],
) -> None:
    """Append a summary row to runs.csv for this batch execution."""
    if not run_id:
        return

    runs_path = db_root / "runs.csv"
    fieldnames = [
        "run_id",
        "started_at",
        "completed_at",
        "n_success",
        "n_failed",
        "n_timeout",
        "n_skipped",
        "n_not_attempted",
    ]

    from collections import Counter

    counts = Counter(row["status"] for row in rows)
    completed_at = datetime.now(timezone.utc).isoformat()

    run_row = {
        "run_id": run_id,
        "started_at": started_at or "",
        "completed_at": completed_at,
        "n_success": counts.get("success", 0),
        "n_failed": counts.get("failed", 0) + counts.get("failed (legacy)", 0),
        "n_timeout": counts.get("timeout", 0),
        "n_skipped": counts.get("skipped", 0),
        "n_not_attempted": counts.get("not_attempted", 0),
    }

    write_header = not runs_path.exists()
    with open(runs_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(run_row)
    print(f"Appended run summary to {runs_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate dataset status.json files into run_status.csv"
    )
    parser.add_argument("db_root", type=Path, help="Path to routee-transit-db root")
    parser.add_argument(
        "--run_id", type=str, default=None, help="Run ID to record in runs.csv"
    )
    parser.add_argument(
        "--started_at",
        type=str,
        default=None,
        help="ISO timestamp when the run started",
    )
    args = parser.parse_args()

    rows = aggregate(args.db_root)
    write_run_status_csv(rows, args.db_root)
    append_runs_csv(args.db_root, args.run_id, args.started_at, rows)


if __name__ == "__main__":
    main()
