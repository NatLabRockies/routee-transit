"""
This script runs the full RouteE-Transit energy prediction pipeline for a single transit
agency.

The pipeline performs the following steps:
1.  Reads in GTFS data optionally filters down to a subset of trips based on date and
    route names. Also optionally adds estimated deadhead trips.
2.  Processes the GTFS data to estimate speed and elevation along the set of road links
    traveled in each bus trip. This is done by (1) map-matching GTFS shapes to the
    OpenStreetMap road network with mappymatch, (2) aggregating map data to the link
    level, and (3) using gradeit to get elevation data.
3.  Uses RouteE vehicle model(s) to predict energy consumption for each roadway link
    and saves this data at both the (detailed) link and (aggregated) trip level.

Outputs are saved as .csv files in the specified output directory.
"""

if __name__ == "__main__":
    import logging
    import os
    import time
    import warnings
    from pathlib import Path

    from routee.transit import GTFSEnergyPredictor

    # Suppress GDAL/PROJ warnings
    os.environ["PROJ_DEBUG"] = "0"
    # Suppress pandas FutureWarning from RouteE-Powertrain
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*swapaxes.*")

    # Configure logging
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    logger = logging.getLogger("single_agency_analysis")

    HERE = Path(__file__).parent.resolve()

    # Configuration
    n_proc = 8
    routee_vehicle_models = [
        "Transit_Bus_Battery_Electric",
        "Transit_Bus_Diesel",
    ]
    input_directory = HERE / "../sample-inputs/saltlake/gtfs"
    output_directory = HERE / "../reports/saltlake"

    start_time = time.time()

    predictor = GTFSEnergyPredictor(
        gtfs_path=input_directory,
        n_processes=n_proc,
        vehicle_models=routee_vehicle_models,
    )

    # Run entire pipeline with one method call
    results = predictor.run(
        date="2023/08/02",
        routes=["205"],
        add_mid_block_deadhead=True,
        add_depot_deadhead=True,
        add_hvac=True,
        output_dir=output_directory,
        save_results=True,
    )

    logger.info(f"Predicted energy for {len(results)} trips")
