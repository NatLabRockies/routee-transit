"""
This script runs the full RouteE-Transit energy prediction pipeline for a single transit
agency.

It demonstrates two approaches:
1. Simple approach: Use the high-level run() method (recommended for most users)
2. Detailed approach: Chain individual methods for fine-grained control

The pipeline performs the following steps:
1.  Reads in GTFS data into a gtfsblocks.Feed object and optionally filters down to a
    subset of trips based on date and route names.
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

    from nrel.routee.transit import GTFSEnergyPredictor

    # Suppress GDAL/PROJ warnings
    os.environ["PROJ_DEBUG"] = "0"
    # Suppress pandas FutureWarning from mappymatch
    warnings.filterwarnings("ignore", category=FutureWarning, module="mappymatch")

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
        # Add more models here to run predictions for multiple vehicles
    ]
    add_thermal_impacts = True
    input_directory = HERE / "../sample-inputs/saltlake/gtfs"
    depot_directory = HERE / "../FTA_Depot"
    output_directory = HERE / "../reports/saltlake"

    # ========================================================================
    # APPROACH 1: Simple - Use the high-level run() method (RECOMMENDED)
    # ========================================================================
    logger.info("=" * 70)
    logger.info("APPROACH 1: Using simple run() method")
    logger.info("=" * 70)

    start_time = time.time()

    predictor = GTFSEnergyPredictor(
        gtfs_path=input_directory,
        depot_path=depot_directory,
        n_processes=n_proc,
    )

    # Run entire pipeline with one method call
    results = predictor.run(
        vehicle_models=routee_vehicle_models,
        date="2023/08/02",
        routes=["205"],
        add_between_trip_deadhead=True,
        add_depot_deadhead=True,
        add_grade=True,
        add_hvac=add_thermal_impacts,
        output_dir=output_directory,
        save_results=True,
    )

    logger.info(f"Approach 1 completed in {time.time() - start_time:.2f} s")
    logger.info(f"Predicted energy for {len(results)} trips")

    # ========================================================================
    # APPROACH 2: Detailed - Chain individual methods for fine-grained control
    # ========================================================================
    # Uncomment below to see the detailed approach

    logger.info("=" * 70)
    logger.info("APPROACH 2: Using detailed method chaining")
    logger.info("=" * 70)

    start_time = time.time()

    predictor = (
        GTFSEnergyPredictor(
            gtfs_path=input_directory,
            depot_path=depot_directory,
            n_processes=n_proc,
        )
        .load_gtfs_data()
        .filter_trips(date="2023/08/02", routes=["205"])
        .add_between_trip_deadhead()
        .add_depot_deadhead()
        .match_shapes_to_network()
        .add_road_grade()
    )

    predictor.predict_energy(
        vehicle_models=routee_vehicle_models,
        add_hvac=add_thermal_impacts,
    )

    predictor.save_results(output_directory)

    logger.info(f"Approach 2 completed in {time.time() - start_time:.2f} s")
