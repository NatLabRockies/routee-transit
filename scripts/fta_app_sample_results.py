"""
This script runs the full RouteE-Transit energy prediction pipeline for a few agencies
to provide sample inputs for the FTA app.
"""

import geopandas as gpd
from gtfsblocks import Feed
from shapely.geometry import LineString


def build_routes_gdf(feed: Feed) -> gpd.GeoDataFrame:
    """Build GeoDataFrame of typical shapes for all routes in a Feed."""
    route_shape_cts = (
        feed.trips.groupby(["route_id", "shape_id"])["trip_count"].sum().reset_index()
    )
    most_common_shape = route_shape_cts.loc[
        route_shape_cts.groupby("route_id")["trip_count"].idxmax()
    ].reset_index(drop=True)

    route_cols = ["route_short_name"]
    if "route_color" in feed.routes.columns:
        route_cols += ["route_color"]
    most_common_shape = most_common_shape.merge(
        feed.routes[route_cols],
        left_on="route_id",
        right_index=True,
    )

    # Build a LineString geometry for each selected shape_id from shapes.txt
    shape_ids = most_common_shape["shape_id"].unique()
    shapes_df = feed.shapes[feed.shapes["shape_id"].isin(shape_ids)].copy()

    # Sort shape points in sequence order, then build one LineString per shape_id
    shape_lines = (
        shapes_df.sort_values(["shape_id", "shape_pt_sequence"])
        .groupby("shape_id")
        .apply(
            lambda g: LineString(zip(g["shape_pt_lon"], g["shape_pt_lat"])),
            include_groups=False,
        )
        .rename("geometry")
        .reset_index()
    )

    # Join geometry back to the most-common-shape table
    gdf = most_common_shape.merge(shape_lines, on="shape_id", how="left")
    gdf = gdf.drop(columns="trip_count")
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    return gdf


if __name__ == "__main__":
    import logging
    import os
    import time
    import warnings

    import pandas as pd

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
        "Transit_Bus_Battery_Electric",
        "Transit_Bus_Diesel",
    ]

    # TODO: aggregate feeds in this script as well
    # example data built with pixi run -e dev-py311 python scripts/feeds/gather_feeds.py --feed_ids mdb-292 mdb-2432 mdb-2303 --db_root frontend_example
    db_root = package_root().parents[1] / "frontend_example"
    feeds_path = db_root / "feeds.csv"
    datasets_path = db_root / "datasets.csv"

    feeds = pd.read_csv(feeds_path)
    # datasets = pd.read_csv(datasets_path)

    feeds_incl = feeds["id"].tolist()
    datasets_incl = feeds["latest_dataset_id"].tolist()  # assumes no validation errors

    for ix, d_id in enumerate(datasets_incl):
        # TODO: export routes geojson
        input_directory = db_root / d_id / "gtfs"
        output_directory = db_root / d_id / "results"

        start_time = time.time()

        predictor = GTFSEnergyPredictor(
            gtfs_path=input_directory,
            n_processes=n_proc,
            vehicle_models=routee_vehicle_models,
            output_dir=output_directory,
            feed_id=feeds_incl[ix],
            dataset_id=d_id,
        )

        # Run entire pipeline
        results = predictor.run(
            date=None,
            routes=None,
            add_mid_block_deadhead=True,
            add_depot_deadhead=True,
            add_hvac=True,
            save_results=True,
        )
        logger.info(f"Predicted energy for {len(results)} trips in {d_id}")

        # Export routes to GeoJSON
        routes_gdf = build_routes_gdf(predictor.feed)
        routes_gdf.to_file(output_directory / "routes.geojson", driver="GeoJSON")
        logger.info(f"Wrote {len(routes_gdf)} route shapes")
