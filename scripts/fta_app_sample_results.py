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
    n_proc = 8
    routee_vehicle_models = [
        "Transit_Bus_Electric_40ft_300kWh",
        "Transit_Bus_Electric_60ft_600kWh",
        "Transit_Bus_Diesel_40ft",
        "Transit_Bus_Diesel_60ft",
    ]

    # Set to True to (re)download the GTFS feeds before running predictions
    gather_feeds = False
    if gather_feeds:
        import subprocess

        subprocess.run(
            [
                "python",
                "scripts/feeds/gather_feeds.py",
                "--db_root",
                "reports/fta_results_100626",
                "--feed_ids",
                "mdb-179",
                "mdb-205",
                "mdb-247",
                "mdb-267",
            ]
        )

    db_root = package_root().parents[1] / "reports" / "fta_results_100626"
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
            overwrite=False,
        )

        # Run entire pipeline across all service dates
        results = predictor.run(
            date=None,
            routes=None,
            add_mid_block_deadhead=True,
            add_depot_deadhead=True,
            add_hvac=True,
            scale_to_year=True,
        )

        logger.info(
            "Finished %d daily predictions for %s in %.1f seconds; results are in %s",
            len(results),
            d_id,
            time.time() - start_time,
            output_directory,
        )
