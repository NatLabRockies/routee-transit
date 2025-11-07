import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from shapely.geometry import Point


def create_depot_deadhead_trips(trips_df: pd.DataFrame) -> pd.DataFrame:
    """Create deadhead trips from and to depots for each block.

    This function essentially creates rows for the trips.txt DataFrame.
    It does not generate shape traces for them (that is handled by other
    functions in this module).

    Parameters
    ----------
    trips_df : pd.DataFrame
        trips_df of selected date route (e.g. result from read_in_gtfs).

    Returns
    -------
    pd.DataFrame: DataFrame with created deadhead trips.
    """

    existing_trips_df = trips_df
    block_ids = existing_trips_df["block_id"].dropna().unique().tolist()
    # For each block id, create two deadhead trips: one from depot to first stop,
    # and one from last stop to depot.
    to_depot_trips = pd.DataFrame(
        {
            "trip_id": [],
            "route_id": [],
            "service_id": [],
            "block_id": [],
            "shape_id": [],
            "route_short_name": [],
            "route_type": [],
            "route_desc": [],
            "agency_id": [],
        }
    )
    from_depot_trips = pd.DataFrame(
        {
            "trip_id": [],
            "route_id": [],
            "service_id": [],
            "block_id": [],
            "shape_id": [],
            "route_short_name": [],
            "route_type": [],
            "route_desc": [],
            "agency_id": [],
        }
    )
    for block_id in block_ids:
        block_trips = existing_trips_df[existing_trips_df["block_id"] == block_id]
        # Exclude any between-trip deadhead trips that may have been added
        block_trips = block_trips.loc[block_trips["from_trip"].isna()]
        # TODO: ensure trips have been sorted in chronological error
        first_trip = block_trips.iloc[0]
        last_trip = block_trips.iloc[-1]
        # Create trip from depot to first stop
        from_depot_trip_id = f"depot_to_{first_trip['trip_id']}"
        from_depot_trip = {
            "trip_id": from_depot_trip_id,
            "route_id": first_trip["route_id"],
            "service_id": first_trip["service_id"],
            "block_id": block_id,
            "shape_id": f"from_depot_{block_id}",
            "route_short_name": f"from_depot_{block_id}",
            "route_type": 3,  # 3 means bus
            "route_desc": f"Deadhead from depot to {first_trip['trip_id']}",
            "agency_id": first_trip.get("agency_id", None),
        }
        from_depot_trips = pd.concat(
            [from_depot_trips, pd.DataFrame([from_depot_trip])], ignore_index=True
        )
        # Create trip from last stop to depot
        to_depot_trip_id = f"{last_trip['trip_id']}_to_depot"
        to_depot_trip = {
            "trip_id": to_depot_trip_id,
            "route_id": last_trip["route_id"],
            "service_id": last_trip["service_id"],
            "block_id": block_id,
            "shape_id": f"to_depot_{block_id}",
            "route_short_name": last_trip.get("route_short_name", ""),
            "route_type": last_trip.get("route_type", 3),  # Default to bus
            "route_desc": f"Deadhead from {last_trip['trip_id']} to depot",
            "agency_id": last_trip.get("agency_id", None),
        }
        to_depot_trips = pd.concat(
            [to_depot_trips, pd.DataFrame([to_depot_trip])], ignore_index=True
        )

    deadhead_trips_df = pd.concat([from_depot_trips, to_depot_trips], ignore_index=True)
    return deadhead_trips_df


def infer_depot_trip_endpoints(
    trips_df: pd.DataFrame, feed: Any, path_to_depots: str | Path
) -> tuple[Any, Any]:
    """Add origin/destination depot geometry for each block.

    Parameters
    ----------
    trips_df: pd.DataFrame
        trips_df of selected date and route (result from read_in_gtfs).
    feed : Any
        GTFS feed object (e.g. result from read_in_gtfs).
    path_to_depots : str | Path
        Path to a vector file (GeoJSON/Shapefile) containing depot point geometries.

    Returns
    -------
    tuple[GeoDataFrame, GeoDataFrame]
        (first_stops_gdf, last_stops_gdf). Each GeoDataFrame contains the stop
        geometry (column 'stop_geometry') and the matched depot geometry
        (column 'depot_geometry').
    """

    # Process trips and stops dataframes in feed to get first and last stops of each block id
    trips_df = trips_df.copy()
    stop_times_df = feed.stop_times
    stops_df = feed.stops
    blocks_trips_stops = stop_times_df.merge(
        trips_df[["trip_id", "block_id"]], on="trip_id", how="right"
    )
    blocks_trips_stops = blocks_trips_stops.merge(stops_df, on="stop_id", how="left")

    blocks_trips_stops = blocks_trips_stops.sort_values(by=["block_id", "arrival_time"])
    first_stops = blocks_trips_stops.groupby("block_id").first().reset_index()
    last_stops = blocks_trips_stops.groupby("block_id").last().reset_index()

    first_stops = first_stops[["block_id", "arrival_time", "stop_lat", "stop_lon"]]
    last_stops = last_stops[["block_id", "arrival_time", "stop_lat", "stop_lon"]]

    first_stops["geometry"] = first_stops.apply(
        lambda row: Point(row["stop_lon"], row["stop_lat"]), axis=1
    )
    last_stops["geometry"] = last_stops.apply(
        lambda row: Point(row["stop_lon"], row["stop_lat"]), axis=1
    )
    first_stops_gdf = gpd.GeoDataFrame(
        first_stops, geometry="geometry", crs="EPSG:4326"
    )
    last_stops_gdf = gpd.GeoDataFrame(last_stops, geometry="geometry", crs="EPSG:4326")

    # Read depot locations; ensure file exists
    if not os.path.exists(path_to_depots):
        raise FileNotFoundError(f"Depot file not found: {path_to_depots}")
    depots_df = gpd.read_file(path_to_depots)
    # Ensure depot geometries are points and in WGS84
    if depots_df.crs is None:
        depots_df = depots_df.set_crs(epsg=4326)
    else:
        depots_df = depots_df.to_crs(epsg=4326)

    # Create a simple mapping from depot index to geometry for fast lookup
    depots_geom_map = depots_df["geometry"].to_dict()

    # Project to Web Mercator (EPSG:3857) for distance computations
    proj_crs = "EPSG:3857"
    first_proj = first_stops_gdf.to_crs(proj_crs).reset_index(drop=True)
    last_proj = last_stops_gdf.to_crs(proj_crs).reset_index(drop=True)
    depots_proj = depots_df.to_crs(proj_crs).copy()

    best_depot_idx = {}
    for block_id, first_row in first_proj.groupby("block_id"):
        first_geom = first_row.iloc[0].geometry
        last_geom = last_proj.loc[last_proj["block_id"] == block_id, "geometry"].values[
            0
        ]

        # Compute pull-out, pull-in, and total distances
        depots_proj["pullout"] = depots_proj.geometry.distance(first_geom)
        depots_proj["pullin"] = depots_proj.geometry.distance(last_geom)
        depots_proj["total"] = depots_proj["pullout"] + depots_proj["pullin"]

        best_idx = depots_proj["total"].idxmin()
        best_depot_idx[block_id] = best_idx

    first_stops_gdf["nearest_depot_idx"] = first_stops_gdf["block_id"].map(
        best_depot_idx
    )
    last_stops_gdf["nearest_depot_idx"] = last_stops_gdf["block_id"].map(best_depot_idx)

    first_stops_gdf["geometry_origin"] = first_stops_gdf["nearest_depot_idx"].map(
        depots_geom_map
    )
    first_stops_gdf["geometry_destination"] = first_stops_gdf.geometry
    last_stops_gdf["geometry_destination"] = last_stops_gdf["nearest_depot_idx"].map(
        depots_geom_map
    )
    last_stops_gdf["geometry_origin"] = last_stops_gdf.geometry

    # Set the arrival time as departure time for deadhead trip to depot for the last_stop_gdf
    last_stops_gdf["departure_time"] = last_stops_gdf["arrival_time"]
    # Drop the arrival_time column for the last_stop_gdf
    last_stops_gdf = last_stops_gdf.drop(columns=["arrival_time"])

    # Keep only relevant columns and set stop_geometry as the active geometry
    first_stops_gdf = first_stops_gdf.drop(columns=["geometry"])
    first_stops_gdf = gpd.GeoDataFrame(
        first_stops_gdf, geometry="geometry_destination", crs="EPSG:4326"
    )

    last_stops_gdf = last_stops_gdf.drop(columns=["geometry"])
    last_stops_gdf = gpd.GeoDataFrame(
        last_stops_gdf, geometry="geometry_origin", crs="EPSG:4326"
    )

    return first_stops_gdf, last_stops_gdf


def create_depot_deadhead_stops(
    first_stops_gdf: gpd.GeoDataFrame,
    last_stops_gdf: gpd.GeoDataFrame,
    deadhead_trips: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create stop_times and stops for deadhead trips from and to depots.

    Parameters
    ----------
    first_stops_gdf: gpd.GeoDataFrame
        GeoDataFrame of first stops for each block id with depot geometry (result from add_depot_to_blocks.py).
    last_stops_gdf: gpd.GeoDataFrame
        GeoDataFrame of last stops for each block id with depot geometry (result from add_depot_to_blocks.py).
    deadhead_trips: pd.DataFrame
        deadhead trip results from create_depot_deadhead_trips.py.
    Returns
    -------
    pd.DataFrame
        DataFrame of stop_times and stops for the deadhead trips.
    """

    from_depot = first_stops_gdf.copy()
    to_depot = last_stops_gdf.copy()

    # Calculate distance from depot to first stop
    from_depot["distance_m"] = from_depot.apply(
        lambda row: geodesic(
            (row.geometry_origin.y, row.geometry_origin.x),
            (row.geometry_destination.y, row.geometry_destination.x),
        ).meters,
        axis=1,
    )
    # Calculate distance from last stop to depot
    to_depot["distance_m"] = to_depot.apply(
        lambda row: geodesic(
            (row.geometry_origin.y, row.geometry_origin.x),
            (row.geometry_destination.y, row.geometry_destination.x),
        ).meters,
        axis=1,
    )
    # Assume average speed of 30 km/h (to be consistant with the number adopted in gtfs_feature_processing.py)
    # to estimate travel time
    from_depot["travel_time_sec"] = (from_depot["distance_m"] / 30000) * 3600
    to_depot["travel_time_sec"] = (to_depot["distance_m"] / 30000) * 3600
    # Calculate departure time from depot for deadhead trip to first stop
    from_depot["departure_time"] = from_depot["arrival_time"] - pd.to_timedelta(
        from_depot["travel_time_sec"], unit="s"
    )
    # Calculate arrival time at depot for deadhead trip from last stop
    to_depot["arrival_time"] = to_depot["departure_time"] + pd.to_timedelta(
        to_depot["travel_time_sec"], unit="s"
    )

    # Create stop_times df for deadhead trips
    deadhead_trips_df = deadhead_trips.copy()
    stop_times_df = pd.DataFrame(
        columns=[
            "trip_id",
            "stop_sequence",
            "arrival_time",
            "stop_id",
            "departure_time",
            "shape_dist_traveled",
        ]
    )
    stop_times_df["trip_id"] = deadhead_trips_df["trip_id"].repeat(2).values
    stop_times_df["stop_sequence"] = [1, 2] * len(deadhead_trips_df)
    stop_times_df["arrival_time"] = [
        x
        for pair in zip(
            from_depot["departure_time"].to_list(), from_depot["arrival_time"].to_list()
        )
        for x in pair
    ] + [
        x
        for pair in zip(
            to_depot["departure_time"].to_list(), to_depot["arrival_time"].to_list()
        )
        for x in pair
    ]
    stop_times_df["stop_id"] = range(1, len(stop_times_df) + 1)
    stop_times_df["stop_id"] = stop_times_df["stop_id"].apply(
        lambda x: f"depot_deadhead_{x}"
    )
    stop_times_df["departure_time"] = stop_times_df["arrival_time"]
    stop_times_df["shape_dist_traveled"] = 0.0

    # Create stops df for deadhead trips
    stops_df = pd.DataFrame(columns=["stop_id", "stop_lat", "stop_lon"])
    stops_df["stop_id"] = stop_times_df["stop_id"]

    x_start_from_depot = from_depot.geometry_origin.apply(lambda p: p.x).to_numpy()
    x_end_from_depot = from_depot.geometry_destination.apply(lambda p: p.x).to_numpy()
    x_start_to_depot = to_depot.geometry_origin.apply(lambda p: p.x).to_numpy()
    x_end_to_depot = to_depot.geometry_destination.apply(lambda p: p.x).to_numpy()
    stop_lon_from_depot = np.ravel(
        np.column_stack((x_start_from_depot, x_end_from_depot))
    )
    stop_lon_to_depot = np.ravel(np.column_stack((x_start_to_depot, x_end_to_depot)))

    y_start_from_depot = from_depot.geometry_origin.apply(lambda p: p.y).to_numpy()
    y_end_from_depot = from_depot.geometry_destination.apply(lambda p: p.y).to_numpy()
    y_start_to_depot = to_depot.geometry_origin.apply(lambda p: p.y).to_numpy()
    y_end_to_depot = to_depot.geometry_destination.apply(lambda p: p.y).to_numpy()
    stop_lat_from_depot = np.ravel(
        np.column_stack((y_start_from_depot, y_end_from_depot))
    )
    stop_lat_to_depot = np.ravel(np.column_stack((y_start_to_depot, y_end_to_depot)))

    stops_df["stop_lat"] = list(stop_lat_from_depot) + list(stop_lat_to_depot)
    stops_df["stop_lon"] = list(stop_lon_from_depot) + list(stop_lon_to_depot)

    return stop_times_df, stops_df
