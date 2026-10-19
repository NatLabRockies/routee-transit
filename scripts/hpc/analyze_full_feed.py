"""Run the current RouteE-Transit model for one Mobility Database dataset."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

import pandas as pd

from routee.transit import GTFSEnergyPredictor
from routee.transit.speed_model import DEFAULT_SPEED_MODEL_DIR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Shared with the FTA sample script so both emit the same route/energy schema.
from fta_app_sample_results import build_routes_gdf  # noqa: E402

VEHICLE_MODELS = [
    "Transit_Bus_Electric_40ft_300kWh",
    "Transit_Bus_Electric_60ft_600kWh",
    "Transit_Bus_Diesel_40ft",
    "Transit_Bus_Diesel_60ft",
    "Transit_Bus_Hybrid_40ft",
    "Transit_Bus_Hybrid_60ft",
    "Transit_Bus_CNG_40ft",
    "Transit_Bus_CNG_60ft",
]


class DatasetTimeout(Exception):
    """Raised when a dataset exceeds its configured wall-clock limit."""


def average_across_dates(results: pd.DataFrame) -> pd.DataFrame:
    """Collapse the date dimension to one row per (trip, vehicle).

    Energy and efficiency are averaged over the dates a trip runs; columns that
    only vary by date are dropped. ``trip_count`` records how many dates each
    trip ran, so annual totals stay recoverable from the averaged table.
    """
    date_cols = {"date", "hvac_energy_kWh", "trip_is_within_gtfs_scope"}
    avg_cols = [c for c in results.columns if c in {"energy_used", "mpge"}]
    group_cols = [c for c in results.columns if c not in date_cols | set(avg_cols)]

    if "date" not in results.columns:
        averaged = results.copy()
        averaged["trip_count"] = 1
        averaged["scenario"] = "median"
        return averaged

    trip_counts = (
        results.groupby("trip_id").agg(trip_count=("date", "nunique")).reset_index()
    )
    averaged = results.groupby(group_cols, as_index=False, dropna=False)[
        avg_cols
    ].mean()
    averaged = averaged.merge(trip_counts, on="trip_id", how="left")
    averaged["scenario"] = "median"
    return averaged


def calculate_mpge(row: pd.Series) -> float:
    if row["energy_unit"] == "kWh":
        return float(row["miles"] / (row["energy_used"] / 33.7))
    if row["energy_unit"] == "gallons_diesel":
        return float(row["miles"] / (row["energy_used"] * 1.136))
    raise ValueError(f"Unsupported energy unit: {row['energy_unit']}")


def get_t3co_inputs(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trip predictions by vehicle and block for T3CO."""
    block_predictions = predictions.dropna(subset=["block_id"]).copy()
    if block_predictions.empty:
        raise ValueError("No block IDs are available for T3CO aggregation")

    group_columns = ["vehicle", "block_id"]
    if "scenario" in block_predictions.columns:
        group_columns.append("scenario")
    grouped = block_predictions.groupby(group_columns, dropna=False)
    result = grouped[["energy_used", "miles"]].sum()
    result["energy_unit"] = grouped["energy_unit"].first()
    result["mpge"] = result.apply(calculate_mpge, axis=1)
    result["range"] = result.apply(
        lambda row: float(
            (500 if row["energy_unit"] == "kWh" else 100)
            / (row["energy_used"] / row["miles"])
        ),
        axis=1,
    )
    return result.reset_index()


def classify_failure(error: Exception) -> str:
    message = str(error).lower()
    if "overpass" in message or "openstreetmap" in message:
        return "overpass_error"
    if "map match" in message or "mappymatch" in message:
        return "map_matching_error"
    return "analysis_error"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict energy consumption for all trips in one GTFS feed."
    )
    parser.add_argument("db_root", type=Path)
    parser.add_argument("dataset", type=str)
    parser.add_argument("--n_processes", type=int, default=mp.cpu_count())
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--speed_model_dir", type=Path, default=None)
    args = parser.parse_args()

    dataset_id = args.dataset
    input_dir = args.db_root / dataset_id / "gtfs"
    output_dir = args.db_root / dataset_id / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.time()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger = logging.getLogger("analyze_full_feed")

    def write_status(
        status: str,
        exit_code: int,
        error: BaseException | None = None,
        failure_type: str | None = None,
    ) -> None:
        status_data = {
            "dataset_id": dataset_id,
            "status": status,
            "run_id": args.run_id,
            "exit_code": exit_code,
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_minutes": round((time.time() - start_time) / 60, 2),
            "includes_deadhead": includes_deadhead,
            "failure_type": failure_type,
            "exception_type": type(error).__name__ if error else None,
            "error_summary": str(error) if error else None,
        }
        (output_dir / "status.json").write_text(json.dumps(status_data, indent=2))

    def timeout_handler(signum: int, frame: object) -> None:
        raise DatasetTimeout(
            f"Analysis exceeded {args.timeout} minute limit for {dataset_id}"
        )

    def terminate_worker(exit_code: int) -> NoReturn:
        if args.timeout and args.timeout > 0:
            signal.alarm(0)
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)

    if args.timeout and args.timeout > 0:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timeout * 60)

    speed_model_dir = args.speed_model_dir or DEFAULT_SPEED_MODEL_DIR
    logger.info("Using speed model directory: %s", speed_model_dir)
    includes_deadhead = False

    try:
        predictor = GTFSEnergyPredictor(
            gtfs_path=input_dir,
            n_processes=args.n_processes,
            vehicle_models=VEHICLE_MODELS,
            output_dir=output_dir,
            dataset_id=dataset_id,
            speed_model_dir=speed_model_dir,
            overwrite=False,
        )
        includes_deadhead = True
        try:
            predictions = predictor.run(
                add_mid_block_deadhead=True,
                add_depot_deadhead=True,
                add_hvac=True,
                scale_to_year=True,
                save_results=False,
            )
        except (DatasetTimeout, KeyboardInterrupt):
            raise
        except Exception as error:
            # Cached networks were built around revenue shapes only, so depots
            # and deadhead paths can fall outside the graph. Revenue-only
            # results are still worth having.
            logger.warning(
                "Deadhead routing failed for %s (%s: %s); retrying without it",
                dataset_id,
                type(error).__name__,
                str(error)[:200],
            )
            includes_deadhead = False
            predictor = GTFSEnergyPredictor(
                gtfs_path=input_dir,
                n_processes=args.n_processes,
                vehicle_models=VEHICLE_MODELS,
                output_dir=output_dir,
                dataset_id=dataset_id,
                speed_model_dir=speed_model_dir,
                overwrite=False,
            )
            predictions = predictor.run(
                add_mid_block_deadhead=False,
                add_depot_deadhead=False,
                add_hvac=True,
                scale_to_year=True,
                save_results=False,
            )

        if predictions.empty:
            logger.info("No valid trips to analyze")
            write_status("skipped", 0)
            return 0

        predictions.to_csv(
            output_dir / "trip_energy_predictions_daily.csv", index=False
        )
        averaged = average_across_dates(predictions)
        averaged.to_csv(output_dir / "trip_energy_predictions.csv", index=False)

        try:
            get_t3co_inputs(predictions).to_csv(
                output_dir / "t3co_inputs.csv", index=False
            )
        except ValueError as error:
            logger.info("Skipping T3CO output: %s", error)

        try:
            build_routes_gdf(predictor.feed, averaged).to_file(
                output_dir / "routes.geojson", driver="GeoJSON"
            )
        except Exception as error:
            logger.warning("Failed to generate routes.geojson: %s", error)

        write_status("success", 0)
        logger.info(
            "Completed %s with %d daily / %d averaged predictions",
            dataset_id,
            len(predictions),
            len(averaged),
        )
        terminate_worker(0)
    except DatasetTimeout as error:
        logger.error("%s", error)
        write_status("timeout", 124, error, "timeout")
        terminate_worker(124)
    except KeyboardInterrupt as error:
        logger.error("Analysis interrupted for %s", dataset_id)
        write_status("failed", 130, error, "interrupted")
        terminate_worker(130)
    except Exception as error:
        failure_type = classify_failure(error)
        logger.exception("Analysis failed for %s (%s)", dataset_id, failure_type)
        exit_code = 75 if failure_type.startswith("overpass_") else 1
        write_status(
            "failed",
            exit_code,
            error,
            failure_type,
        )
        terminate_worker(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
