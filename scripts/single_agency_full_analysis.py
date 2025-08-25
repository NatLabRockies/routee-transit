"""
This script runs the full RouteE-Transit energy prediction pipeline for a single transit
agency.

It performs the following steps:
1.  Reads in GTFS data into a gtfsblocks.Feed object and optionally filters down to a
    subset of trips based on date and route names.
2.  Processes the GTFS data to estimate speed and elevation along the set of road links
    traveled in each bus trip. This is done by (1) map-matching GTFS shapes to the
    OpenStreetMap road network with mappymatch, (2) aggregating map data to the link
    level, and (3) using gradeit to get elevation data. Note that a path to local
    elevation data must be provided (defaults to data/usgs_elevation).
3.  Uses a RouteE vehicle model to predict energy consumption for each roadway link
    and saves this data at both the (detailed) link and (aggregated) trip level. The
    RouteE inputs are saved as well to support making repeated predictions without
    running the entire pipeline over again.

Inputs can be adjusted manually within the script below and are as follows:
-   agency (str): The name of the transit agency to analyze, corresponding to a set of
    GTFS files under data/gtfs
-   routee_vehicle_model (str): The file path to the RouteE vehicle model JSON.
-   raster_path (str): The directory path containing elevation raster data.

Outputs are saved as .csv files in the reports/ directory:
-   reports/energy_predictions/{agency}_link_energy_predictions.csv: link-level energy
    consumption predictions
-   reports/energy_predictions/{agency}_trip_energy_predictions.csv: trip-level energy
    consumption predictions.
"""

if __name__ == "__main__":
    import logging
    import multiprocessing as mp
    import os
    import time
    import pandas as pd
    from pathlib import Path

    from nrel.routee.transit import (
        build_routee_features_with_osm,
        predict_for_all_trips,
        aggregate_results_by_trip,
    )

    # Suppress GDAL/PROJ warnings, which flood the output when we run gradeit
    # TODO: resolve underlying issue that generates these warnings
    os.environ["PROJ_DEBUG"] = "0"

    # Set up logging: Clear any existing handlers
    logging.getLogger().handlers.clear()

    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

    logger = logging.getLogger("single_agency_analysis")

    HERE = Path(__file__).parent.resolve()

    # Set inputs
    n_proc = mp.cpu_count()
    routee_vehicle_model = "Transit_Bus_Diesel"
    input_directory = HERE / "../sample-inputs/saltlake/gtfs"
    output_directory = HERE / "../reports/saltlake"
    if not output_directory.exists():
        output_directory.mkdir(parents=True)

    start_time = time.time()
    routee_input_df = build_routee_features_with_osm(
        input_directory=input_directory,
        date_incl="2023/08/02",
        routes_incl=["9"],
        add_road_grade=True,
        n_processes=n_proc,
    )

    logger.info("Finished building RouteE features")
    # Save geometry data separate from energy predictions to save space
    geom = pd.concat([routee_input_df["road_id"], routee_input_df.pop("geom")], axis=1)
    geom = geom.drop_duplicates(subset="geom")
    geom.to_csv(output_directory / "link_geometry.csv", index=False)

    # Predict energy consumption
    routee_results = predict_for_all_trips(
        routee_input_df=routee_input_df,
        routee_vehicle_model=routee_vehicle_model,
        n_processes=n_proc,
    )
    routee_results["vehicle"] = routee_vehicle_model
    routee_results.to_csv(output_directory / "link_energy_predictions.csv", index=False)

    # Summarize predictions by trip
    energy_by_trip = aggregate_results_by_trip(routee_results, routee_vehicle_model)

    energy_by_trip.to_csv(output_directory / "trip_energy_predictions.csv")
    logger.info(f"Finished predictions in {time.time() - start_time:.2f} s")
