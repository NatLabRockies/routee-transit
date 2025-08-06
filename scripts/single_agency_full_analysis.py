"""
This script runs the full RouteE-Transit energy prediction pipeline for a single transit
agency.

It performs the following steps:
1.  Reads in GTFS data into a gtfsblocks.Feed object and optionally filters down to a
    random subset of trips (for faster testing).
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
-   reports/trip_features/{agency}_trip_features.csv: trip features that serve as inputs
    to RouteE
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

    from pathlib import Path

    from nrel.routee.transit.prediction.gtfs_feature_processing import (
        build_routee_features_with_osm,
    )
    from nrel.routee.transit.prediction.routee import predict_for_all_trips

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
    input_directory = HERE / "../sample-inputs/saltlake"
    output_directory = HERE / "../reports/saltlake"
    if not output_directory.exists():
        output_directory.mkdir(parents=True)

    # Number of trips to include in analysis. If None, all will be analyzed.
    n_trips_incl = 100

    start_time = time.time()
    routee_input_df = build_routee_features_with_osm(
        input_directory=input_directory,
        n_trips=n_trips_incl,
        add_road_grade=True,
        n_processes=n_proc,
    )

    logger.info("Finished building RouteE features")
    routee_input_df.to_csv(output_directory / "trip_features.csv", index=False)

    # 4) Predict energy consumption
    routee_results = predict_for_all_trips(
        routee_input_df=routee_input_df,
        routee_vehicle_model=routee_vehicle_model,
        n_processes=n_proc,
    )
    routee_results["vehicle"] = routee_vehicle_model
    routee_results.to_csv(output_directory / "link_energy_predictions.csv", index=False)

    # Summarize predictions by trip
    agg_cols = [c for c in ["gallons", "kWhs"] if c in routee_results.columns]
    energy_by_trip = routee_results.groupby("trip_id").agg(
        {"kilometers": "sum", **{c: "sum" for c in agg_cols}}
    )
    energy_by_trip["miles"] = 0.6213712 * energy_by_trip["kilometers"]
    energy_by_trip["vehicle"] = routee_vehicle_model
    # TODO: save geometry data separate from energy predictions to save space
    energy_by_trip.drop(columns="kilometers").to_csv(
        output_directory / "trip_energy_predictions.csv"
    )
    logger.info(f"Finished predictions in {time.time() - start_time:.2f} s")
