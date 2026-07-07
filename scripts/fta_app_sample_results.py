"""
This script runs the full RouteE-Transit energy prediction pipeline for a few agencies
to provide sample inputs for the FTA app.
"""

import subprocess

import geopandas as gpd
import pandas as pd
from gtfsblocks import Feed
from shapely.geometry import LineString


def build_routes_gdf(feed: Feed, predictions: pd.DataFrame) -> gpd.GeoDataFrame:
    """Build GeoDataFrame of typical shapes for all routes in a Feed."""
    route_shape_cts = (
        feed.trips.groupby(["route_id", "shape_id"])["trip_id"].nunique().reset_index()
    )
    most_common_shape = route_shape_cts.loc[
        route_shape_cts.groupby("route_id")["trip_id"].idxmax()
    ].reset_index(drop=True)

    # Aggregate energy by route and vehicle, then build nested efficiency dicts
    energy_by_route = (
        predictions.groupby(["route_id", "vehicle"])
        .agg(
            energy_used=("energy_used", "sum"),
            miles=("miles", "sum"),
            energy_unit=("energy_unit", "first"),
            n_trips=("energy_used", "count"),
        )
        .reset_index()
    )

    def _build_energy_dicts(group: pd.DataFrame) -> pd.Series:
        mean_energy_used: dict[str, dict] = {}
        efficiency: dict[str, dict] = {}
        for _, row in group.iterrows():
            vehicle = str(row["vehicle"])
            unit = str(row["energy_unit"])
            mean_energy_used[vehicle] = {
                "value": row["energy_used"] / row["n_trips"],
                "unit": unit,
            }
            if unit == "kWh":
                efficiency[vehicle] = {
                    "value": row["energy_used"] / row["miles"],
                    "unit": "kWh/mi",
                }
            else:
                efficiency[vehicle] = {
                    "value": row["miles"] / row["energy_used"],
                    "unit": "mpg_diesel",
                }
        return pd.Series(
            {"mean_energy_used": mean_energy_used, "mean_efficiency": efficiency}
        )

    nested = (
        energy_by_route.groupby("route_id")
        .apply(_build_energy_dicts, include_groups=False)
        .reset_index()
    )
    most_common_shape = most_common_shape.merge(nested, on="route_id", how="left")

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
    gdf = gdf.drop(columns="trip_id")
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

    # example data built with pixi run -e dev-py311 python scripts/feeds/gather_feeds.py --db_root tmp --feed_ids mdb-292 mdb-1330 mdb-2432
    # subprocess.run([
    #     "python", "scripts/feeds/gather_feeds.py",
    #     "--db_root", "reports/fta_demo_070726",
    #     "--feed_ids", "mdb-179", "mdb-205", "mdb-247", "mdb-267"
    # ])

    db_root = package_root().parents[1] / "reports" / "fta_demo_070726"
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
            save_results=False,
            scale_to_year=True,
        )

        # Collapse the date dimension: average energy consumption across all
        # dates for each (trip, vehicle) pair. Columns that vary by date
        # (energy_used, mpge, hvac_energy_kWh, trip_is_within_gtfs_scope) are
        # either averaged or dropped; all other columns are date-invariant.
        _date_cols = {"date", "hvac_energy_kWh", "trip_is_within_gtfs_scope"}
        _avg_cols = [c for c in results.columns if c in {"energy_used", "mpge"}]
        _grp_cols = [c for c in results.columns if c not in _date_cols | set(_avg_cols)]
        trip_counts = results.groupby("trip_id").agg(trip_count=("date", "nunique")).reset_index()
        results = results.groupby(_grp_cols, as_index=False, dropna=False)[_avg_cols].mean()

        # Reintroduce trip_count: number of times each trip runs
        results = results.merge(trip_counts, on="trip_id", how="left")

        results["scenario"] = "median"
        results.to_csv(output_directory / "trip_energy_predictions.csv", index=False)

        logger.info(f"Finished {len(results)} energy predictions for trips in {d_id}")

        # Export routes to GeoJSON
        routes_gdf = build_routes_gdf(predictor.feed, results)
        routes_gdf.to_file(output_directory / "routes.geojson", driver="GeoJSON")
        logger.info(f"Wrote {len(routes_gdf)} route shapes")
