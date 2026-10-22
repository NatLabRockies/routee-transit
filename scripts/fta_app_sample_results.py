"""
This script runs the full RouteE-Transit energy prediction pipeline for a few agencies
to provide sample inputs for the FTA app.
"""

import pandas as pd

if __name__ == "__main__":
    import logging
    import os
    import time
    import warnings

    from routee.transit import GTFSEnergyPredictor, package_root

    # Suppress GDAL/PROJ warnings
    os.environ["PROJ_DEBUG"] = "0"
    # Suppress pandas FutureWarning from RouteE-Powertrain
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*swapaxes.*")

    # Configure logging
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    logger = logging.getLogger("fta_example_db")

    # HERE = Path(__file__).parent.resolve()

    # Configuration
    n_proc = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    routee_vehicle_models = [
        "Transit_Bus_Electric_40ft_300kWh",
        "Transit_Bus_Electric_60ft_600kWh",
        "Transit_Bus_Diesel_40ft",
        "Transit_Bus_Diesel_60ft",
        "Transit_Bus_CNG_40ft",
        "Transit_Bus_CNG_60ft",
        "Transit_Bus_Hybrid_40ft",
        "Transit_Bus_Hybrid_60ft",
    ]

    # The app only consumes the aggregated trip results; these exports are large
    # and only needed for deeper analysis.
    save_daily_results = False
    save_link_results = False

    db_root = package_root().parents[1] / "reports" / "fta_results_101926"

    # Set to True to (re)download the GTFS feeds before running predictions
    gather_feeds = True
    if gather_feeds:
        from scripts.feeds.gather_feeds import main as gather_feeds_main

        gather_feeds_main(
            [
                "--db_root",
                str(db_root),
                "--feed_ids",
                "mdb-117",
                "mdb-205",
                "mdb-247",
                "mdb-267",
                "mdb-368",
                "mdb-589",
                "mdb-3521",
            ]
        )

    db_root = package_root().parents[1] / "reports" / "fta_results_100826"
    feeds_path = db_root / "feeds.csv"
    datasets_path = db_root / "datasets.csv"

    feeds = pd.read_csv(feeds_path)
    datasets = pd.read_csv(datasets_path)

    feeds_incl = feeds["id"].tolist()
    datasets_incl = datasets["id"].tolist()  # assumes no validation errors

    for ix, d_id in enumerate(datasets_incl):
        input_directory = db_root / d_id / "gtfs"
        output_directory = db_root / d_id / "results"

        logger.info(f"Starting predictions for {d_id}")
        start_time = time.time()

        predictor = GTFSEnergyPredictor(
            gtfs_path=input_directory,
            n_processes=n_proc,
            vehicle_models=routee_vehicle_models,
            output_dir=output_directory,
            feed_id=feeds_incl[ix],
            dataset_id=d_id,
            overwrite=True,
            explode_block_routes=True,
            explode_block_routes=True,
        )

        # Run entire pipeline across all service dates
        daily_results = predictor.run(
            date=None,
            routes=None,
            add_mid_block_deadhead=True,
            add_depot_deadhead=True,
            add_hvac=True,
            scale_to_year=True,
            save_results=False,
        )
        predictor.save_results(
            save_trip_level=save_daily_results,
            save_link_level=save_link_results,
            save_route_shapes=True,
            save_block_level=True,
        )

        # The FTA app expects one row per (trip, vehicle), so collapse the date
        # dimension: average the date-varying energy columns and record how many
        # days each trip operates. All remaining columns are date-invariant.
        date_varying = {"energy_used", "mpge", "hvac_energy_kWh"}
        dropped = {"date", "trip_is_within_gtfs_scope"}
        avg_cols = [c for c in daily_results.columns if c in date_varying]
        group_cols = [
            c for c in daily_results.columns if c not in date_varying | dropped
        ]
        trip_counts = (
            daily_results.groupby("trip_id")
            .agg(trip_count=("date", "nunique"))
            .reset_index()
        )
        results = daily_results.groupby(group_cols, as_index=False, dropna=False)[
            avg_cols
        ].mean()
        results = results.merge(trip_counts, on="trip_id", how="left")

        if save_daily_results:
            # save_results() already claimed the trip filename for the daily export.
            daily_path = output_directory / "trip_energy_predictions.csv"
            daily_path.replace(output_directory / "trip_energy_by_day.csv")
        results.to_csv(output_directory / "trip_energy_predictions.csv", index=False)

        logger.info(
            "Finished %d aggregated predictions (%d daily) for %s in %.1f seconds; "
            "results are in %s",
            len(results),
            len(daily_results),
            d_id,
            time.time() - start_time,
            output_directory,
        )
