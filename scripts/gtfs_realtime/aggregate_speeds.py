import json
import logging
import multiprocessing as mp
import os
import time
from functools import partial
from itertools import groupby
from pathlib import Path

import folium
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from mappymatch.constructs.geofence import Geofence
from mappymatch.constructs.trace import Trace
from mappymatch.maps.nx.nx_map import NetworkType, NxMap
from mappymatch.matchers.lcss.lcss import LCSSMatcher
from pyproj import Transformer
from scipy.spatial import cKDTree


def read_realtime_records(path_to_json):
    # Read and flatten the JSON data
    records = []
    with open(path_to_json, "r") as f:
        for line in f:
            record = json.loads(line)
            records.append(record)

    # Flatten the nested JSON structure
    df = pd.json_normalize(records)
    column_names = {
        "vehicle.trip.tripId": "trip_id",
        "vehicle.trip.startDate": "start_date",
        "vehicle.trip.startTime": "start_time",
        "vehicle.trip.routeId": "route_id",
        "vehicle.position.latitude": "latitude",
        "vehicle.position.longitude": "longitude",
        "vehicle.currentStopSequence": "current_stop_sequence",
        "vehicle.currentStatus": "current_status",
        "vehicle.timestamp": "timestamp",
        "vehicle.vehicle.id": "vehicle_id",
        "vehicle.vehicle.label": "vehicle_label",
        "vehicle.occupancyStatus": "occupancy_status",
        "vehicle.stopId": "stop_id",
        "vehicle.position.speed": "speed",
        "vehicle.position.bearing": "bearing",
    }
    df = df.rename(columns=column_names)
    
    df["trip_id"] = df["trip_id"].astype(str)
    return df


def clean_trip_df(trip_rt_df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates in realtime DataFrame for a single trip.

    Args:
        trip_rt_df (pd.DataFrame): DataFrame with realtime observations for a single
            bus trip

    Returns:
        pd.DataFrame: cleaned DataFrame with duplicates removed
    """
    trip_rt_df["timestamp"] = pd.to_datetime(
        trip_rt_df["timestamp"].astype(int), unit="s"
    )
    # Remove duplicate observations based on key columns
    keep_cols = [
        "timestamp",
        "latitude",
        "longitude",
        "current_stop_sequence",
        "stop_id",
    ]
    keep_cols = [c for c in keep_cols if c in trip_rt_df.columns]
    trip_rt_filt = trip_rt_df.drop_duplicates(subset=keep_cols).reset_index(drop=True)
    return trip_rt_filt


def match_realtime_trip(trip_rt_df: pd.DataFrame) -> pd.DataFrame:
    """Match realtime observations to the OSM network using mappymatch.

    Args:
        trip_rt_df (pd.DataFrame): DataFrame of GTFS Realtime observations for
            a single bus trip.
    Returns:
        pd.DataFrame: trip_rt_df with road link features appended
    """
    # Create mappymatch trace
    trace = Trace.from_dataframe(
        trip_rt_df, lat_column="latitude", lon_column="longitude"
    )

    # Create geofence and use it to pull network
    geofence = Geofence.from_trace(trace, padding=800)
    nxmap = NxMap.from_geofence(geofence, network_type=NetworkType.DRIVE)
    # Run map matching algorithm
    matcher = LCSSMatcher(nxmap)
    matches = matcher.match_trace(trace).matches_to_dataframe()
    try:
        matched_df = pd.concat(
            [
                trip_rt_df,
                matches[
                    ["road_id", "geom", "distance_to_road", "kilometers", "travel_time"]
                ],
            ],
            axis=1,
        )
    except KeyError as e:
        raise ValueError(
            f"Map matching error. Returned df is {matches.head()}"
        ) from e
    return matched_df


def calculate_shape_distance(shape_df, lat1, lon1, lat2, lon2, max_distance_meters=100):
    """
    Calculate the distance between two points along a shape.

    Parameters:
    -----------
    shape_df : pd.DataFrame
        DataFrame with columns 'shape_pt_lat', 'shape_pt_lon', 'shape_dist_traveled'
        sorted by 'shape_pt_sequence'
    lat1, lon1 : float
        Coordinates of the first point
    lat2, lon2 : float
        Coordinates of the second point
    max_distance_meters : float
        Maximum allowed distance (in meters) from a point to the shape

    Returns:
    --------
    float
        Distance along the shape between the two points (in the units of shape_dist_traveled)

    Raises:
    -------
    ValueError
        If either point is too far from the shape
    """
    # Extract shape coordinates
    shape_coords = shape_df[["shape_pt_lat", "shape_pt_lon"]].values
    shape_distances = shape_df["shape_dist_traveled"].values

    # Build KD-tree for efficient nearest neighbor search
    # Note: For more accurate distance calculations over larger areas,
    # consider projecting to a local coordinate system
    tree = cKDTree(shape_coords)

    # Find nearest points on shape for both input coordinates
    dist1, idx1 = tree.query([lat1, lon1])
    dist2, idx2 = tree.query([lat2, lon2])

    # Convert degrees to approximate meters (rough approximation)
    # At mid-latitudes, 1 degree lat ≈ 111km, 1 degree lon ≈ 111km * cos(lat)
    avg_lat = (lat1 + lat2) / 2
    dist1_meters = (
        dist1 * 111000 * np.sqrt(1 + (np.cos(np.radians(avg_lat))) ** 2) / np.sqrt(2)
    )
    dist2_meters = (
        dist2 * 111000 * np.sqrt(1 + (np.cos(np.radians(avg_lat))) ** 2) / np.sqrt(2)
    )

    # Check if points are within acceptable distance from shape
    if dist1_meters > max_distance_meters:
        logging.debug(
            f"Point 1 ({lat1}, {lon1}) is {dist1_meters:.1f}m from the shape, exceeds max of {max_distance_meters}m"
        )
        return np.nan

    if dist2_meters > max_distance_meters:
        logging.debug(
            f"Point 2 ({lat2}, {lon2}) is {dist2_meters:.1f}m from the shape, exceeds max of {max_distance_meters}m"
        )
        return np.nan

    # Get the shape_dist_traveled values at the matched points
    shape_dist1 = shape_distances[idx1]
    shape_dist2 = shape_distances[idx2]

    # Return the absolute difference (distance along shape)
    return abs(shape_dist2 - shape_dist1)


def add_speed_between_points(
    trip_rt_df: pd.DataFrame, shape_df: pd.DataFrame
) -> pd.DataFrame:
    """Estimate the speed between consecutive points for a single GTFS-RT trip.

    Args:
        trip_rt_df (pd.DataFrame): trip realtime DataFrame
        shape_df (pd.DataFrame): DataFrame of this trip's shape from shapes.txt

    Returns:
        pd.DataFrame: trip_rt_df with distance between observations added
    """
    # Calculate distance between consecutive observations using the shape
    trip_rt_df["shape_distance_to_next"] = np.nan

    for i in range(len(trip_rt_df) - 1):
        lat1 = trip_rt_df.loc[i, "latitude"]
        lon1 = trip_rt_df.loc[i, "longitude"]
        lat2 = trip_rt_df.loc[i + 1, "latitude"]
        lon2 = trip_rt_df.loc[i + 1, "longitude"]

        distance = calculate_shape_distance(
            shape_df, lat1, lon1, lat2, lon2, max_distance_meters=100
        )
        trip_rt_df.loc[i, "shape_distance_to_next"] = distance

    trip_rt_df["timedelta_seconds"] = trip_rt_df["timestamp"].diff().dt.total_seconds()
    FT_TO_MILES = 1.0 / 5280
    trip_rt_df["mph"] = (
        trip_rt_df["shape_distance_to_next"]
        * FT_TO_MILES
        / (trip_rt_df["timedelta_seconds"] / 3600)
    )
    trip_rt_df["mph"] = trip_rt_df["mph"].replace(np.inf, np.nan)
    return trip_rt_df


def add_link_endpoints_to_realtime_df(
    matched_rt_df: pd.DataFrame, geo_crs: str = "EPSG:3857"
) -> pd.DataFrame:
    """Add endpoints of map-matched road links to the realtime data.

    These endpoints are included so that we can interpolate timestamps when the vehicle
    started and finished each road link, in order to estimate average speed over links.

    Args:
        matched_rt_df (pd.DataFrame): Map-matched DataFrame of GTFS-RT observations
            for a single bus trip.
        geo_crs (str, optional): Coordinate system of road link geometry. Defaults to
            "EPSG:3857" (Web Mercator, used by OSM).

    Returns:
        pd.DataFrame: matched_rt_df with rows added for all road link endpoints.
    """
    # Transformer to convert road link shapes to lat/lon degrees
    transformer = Transformer.from_crs(geo_crs, "EPSG:4326", always_xy=True)

    # Identify the start and end coordinates of each map-matched link
    segment_starts = {}
    segment_ends = {}
    for geo in matched_rt_df["geom"].unique():
        coords = shapely.get_coordinates(geo)
        # Transform all coordinates at once
        lon, lat = transformer.transform(coords[:, 0], coords[:, 1])
        segment_starts[geo] = (lat[0], lon[0])
        segment_ends[geo] = (lat[-1], lon[-1])

    # Create a list to hold all rows (existing + new endpoint rows)
    new_rows = []

    # Group by road_id to process each link segment
    for road_id, group in matched_rt_df.groupby("road_id", sort=False):
        # Get the geometry for this road_id (take first since they should all be the same)
        geom = group["geom"].iloc[0]

        # Get start and end coordinates
        start_coords = segment_starts[geom]  # (lat, lon)
        end_coords = segment_ends[geom]  # (lat, lon)

        # Create a row for the segment start with only essential fields
        start_row = pd.Series(index=matched_rt_df.columns, dtype=object)
        start_row["latitude"] = start_coords[0]
        start_row["longitude"] = start_coords[1]
        start_row["is_endpoint"] = "start"
        start_row["geom"] = geom  # Keep geom for reference
        start_row["road_id"] = road_id  # Keep road_id for reference

        # Create a row for the segment end with only essential fields
        end_row = pd.Series(index=matched_rt_df.columns, dtype=object)
        end_row["latitude"] = end_coords[0]
        end_row["longitude"] = end_coords[1]
        end_row["is_endpoint"] = "end"
        end_row["geom"] = geom  # Keep geom for reference
        end_row["road_id"] = road_id  # Keep road_id for reference

        # Add start endpoint, then all original rows, then end endpoint
        new_rows.append(start_row)
        for _, row in group.iterrows():
            row_copy = row.copy()
            row_copy["is_endpoint"] = "original"
            new_rows.append(row_copy)
        new_rows.append(end_row)

    # Create new dataframe with all rows
    return pd.DataFrame(new_rows).reset_index(drop=True)


def interpolate_timestamps(
    df_with_endpoints: pd.DataFrame, shape: pd.DataFrame
) -> pd.DataFrame:
    """Estimate values of missing timestamps for all link start/end points.

    This function estimates the timestamps by interpolation, based on the timestamps
    of the previous and next original points from the GTFS-RT trace and the estimated
    distance from the link endpoint to each of those observed points.

    Args:
        df_with_endpoints (pd.DataFrame): map-matched DataFrame of GTFS-RT trip with
            road link endpoints inserted
        shape (pd.DataFrame): GTFS shape DataFrame

    Returns:
        pd.DataFrame: df_with_endpoints with all timestamps populated
    """

    df_with_endpoints["mph"] = df_with_endpoints["mph"].ffill()
    # Grab coordinates of next point and previous point. We'll calculate the distance to
    # each and interpolate the timestamp based on them.
    # Create a version of coordinates that only includes original observations
    original_mask = df_with_endpoints["is_endpoint"] == "original"
    df_with_endpoints["orig_latitude"] = df_with_endpoints["latitude"].where(
        original_mask
    )
    df_with_endpoints["orig_longitude"] = df_with_endpoints["longitude"].where(
        original_mask
    )
    df_with_endpoints["orig_timestamp"] = df_with_endpoints["timestamp"].where(
        original_mask
    )

    # Forward and backward fill to get previous and next original observations
    df_with_endpoints["last_latitude"] = df_with_endpoints["orig_latitude"].ffill()
    df_with_endpoints["last_longitude"] = df_with_endpoints["orig_longitude"].ffill()
    df_with_endpoints["last_timestamp"] = df_with_endpoints["orig_timestamp"].ffill()

    df_with_endpoints["next_latitude"] = df_with_endpoints["orig_latitude"].bfill()
    df_with_endpoints["next_longitude"] = df_with_endpoints["orig_longitude"].bfill()
    df_with_endpoints["next_timestamp"] = df_with_endpoints["orig_timestamp"].bfill()

    # Calculate distances for endpoint rows that have both previous and next observations
    endpoint_mask = df_with_endpoints["is_endpoint"].isin(["start", "end"])
    # Only process endpoints that have valid last and next coordinates
    valid_interpolation_mask = (
        endpoint_mask
        & df_with_endpoints["last_latitude"].notna()
        & df_with_endpoints["next_latitude"].notna()
    )

    # Distance from last original point to current endpoint
    df_with_endpoints.loc[valid_interpolation_mask, "distance_from_last"] = (
        df_with_endpoints.loc[valid_interpolation_mask].apply(
            lambda row: calculate_shape_distance(
                shape,
                lat1=row["last_latitude"],
                lon1=row["last_longitude"],
                lat2=row["latitude"],
                lon2=row["longitude"],
                max_distance_meters=100,
            ),
            axis=1,
        )
    )

    # Distance from current endpoint to next original point
    df_with_endpoints.loc[valid_interpolation_mask, "distance_to_next"] = (
        df_with_endpoints.loc[valid_interpolation_mask].apply(
            lambda row: calculate_shape_distance(
                shape,
                lat1=row["latitude"],
                lon1=row["longitude"],
                lat2=row["next_latitude"],
                lon2=row["next_longitude"],
                max_distance_meters=100,
            ),
            axis=1,
        )
    )

    # Interpolate timestamp based on distances
    df_with_endpoints.loc[valid_interpolation_mask, "total_distance"] = (
        df_with_endpoints.loc[valid_interpolation_mask, "distance_from_last"]
        + df_with_endpoints.loc[valid_interpolation_mask, "distance_to_next"]
    )

    df_with_endpoints.loc[valid_interpolation_mask, "total_time_seconds"] = (
        df_with_endpoints.loc[valid_interpolation_mask, "next_timestamp"]
        - df_with_endpoints.loc[valid_interpolation_mask, "last_timestamp"]
    ).dt.total_seconds()

    # Interpolate: timestamp = last_timestamp + (distance_from_last / total_distance) * total_time
    df_with_endpoints.loc[valid_interpolation_mask, "timestamp"] = (
        df_with_endpoints.loc[valid_interpolation_mask, "last_timestamp"]
        + pd.to_timedelta(
            (
                df_with_endpoints.loc[valid_interpolation_mask, "distance_from_last"]
                / df_with_endpoints.loc[valid_interpolation_mask, "total_distance"]
            )
            * df_with_endpoints.loc[valid_interpolation_mask, "total_time_seconds"],
            unit="s",
        )
    )
    return df_with_endpoints


def estimate_link_speeds(interpolated_df):
    # Group by road_id to calculate link-level speeds
    link_gb = interpolated_df.groupby("road_id")

    link_summary = pd.DataFrame()

    # Calculate total distance traveled on each link
    # Sum all shape_distance_to_next values, but exclude the last row's value (which is distance to next link)
    link_summary["distance_traveled_km"] = link_gb["kilometers"].mean()

    # Convert to miles
    link_summary["distance_traveled_mi"] = link_summary["distance_traveled_km"] / 1.609

    # Calculate time difference between first and last timestamp on each link
    link_summary["time_diff_sec"] = (
        link_gb["timestamp"].max() - link_gb["timestamp"].min()
    ).dt.total_seconds()

    # Calculate average speed in mph
    link_summary["mph"] = link_summary["distance_traveled_mi"] / (
        link_summary["time_diff_sec"] / 3600
    )

    # Include shape for plotting
    link_summary["geom"] = link_gb["geom"].apply(lambda x: x.bfill().iloc[0])

    # Add some additional useful info
    link_summary["first_timestamp"] = link_gb["timestamp"].min()
    link_summary["n_observations"] = link_gb["is_endpoint"].apply(
        lambda x: (x == "original").sum()
    )
    return link_summary


def get_link_speeds_for_trip(trip_id, rt_df, trips_df, shapes_df):
    # Extract the trip with the most observed points
    trip_df = rt_df[rt_df["trip_id"] == trip_id].copy()

    trip_id = trip_df.trip_id.iloc[0]
    shape_id = trips_df.loc[trip_id, "shape_id"]
    shape = shapes_df[shapes_df["shape_id"] == shape_id].sort_values(
        by="shape_pt_sequence"
    )

    trip_df_filt = clean_trip_df(trip_df)
    if len(trip_df_filt) < 10:
        # Throw out trips with under 10 unique points
        return pd.DataFrame()
    
    trip_with_speeds = add_speed_between_points(trip_df_filt, shape)
    try:
        matched_df = match_realtime_trip(trip_with_speeds)
    except ValueError:
        return pd.DataFrame()
    except shapely.errors.GEOSException:
        return pd.DataFrame()
    
    try:
        matched_df_ext = add_link_endpoints_to_realtime_df(matched_df)
    except TypeError as e:
        print(f"Caught error for trip {trip_id}, skipping. Error was: {e}")
        return pd.DataFrame()

    matched_df_with_timestamps = interpolate_timestamps(matched_df_ext, shape)
    link_summary = estimate_link_speeds(matched_df_with_timestamps)

    # Clean up output
    link_summary.index = link_summary.index.astype(str)
    link_summary["trip_id"] = trip_id
    link_summary["route_id"] = trip_df_filt["route_id"].iloc[0]
    link_summary["vehicle_id"] = trip_df_filt["vehicle_id"].iloc[0]
    return link_summary[
        [
            "geom",
            "mph",
            "trip_id",
            "first_timestamp",
            "vehicle_id",
            "route_id",
            "n_observations",
        ]
    ]


def get_speeds_for_one_day(path_to_json: Path | os.PathLike):
    if not isinstance(path_to_json, Path):
        path_to_json = Path(path_to_json)

    # Parent directory of json file, should contain a "static" folder with
    # GTFS static files (we need trips and shapes)
    gtfs_root = path_to_json.parent

    # Read relevant static files
    try:
        trips_df = pd.read_csv(
            gtfs_root / "static/trips.txt",
            dtype={"trip_id": str, "shape_id": str},
        ).set_index("trip_id")
        shapes_df = pd.read_csv(
            gtfs_root / "static/shapes.txt", dtype={"shape_id": str}
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"GTFS static files not found in expected location: {gtfs_root}/static. "
            "Please ensure 'trips.txt' and 'shapes.txt' exist."
        ) from e

    # Read in realtime data
    rt_df = read_realtime_records(path_to_json)
    print(f"{rt_df.trip_id.nunique()} trips on this day")

    # If route IDs aren't included in realtime data, merge them in from trips.txt.
    if "route_id" not in rt_df.columns:
        rt_df = rt_df.merge(trips_df[["route_id"]], left_on="trip_id", right_index=True)

    first_trip_start = time.time()
    speeds_partial = partial(
        get_link_speeds_for_trip, rt_df=rt_df, shapes_df=shapes_df, trips_df=trips_df
    )

    for ix, (route_id, route_rt) in enumerate(rt_df.groupby("route_id")):
        print(f"Analyzing {route_rt.trip_id.nunique()} trips on route {route_id}")
        mp_inputs = list(route_rt["trip_id"].unique())
        route_start = time.time()
        with mp.Pool(mp.cpu_count() - 2) as pool:
            results = pool.map(speeds_partial, mp_inputs)

        pd.concat(results).to_csv(gtfs_root / f"realtime_speeds_{route_id}.csv")
        print(
            f"Analyzing {route_rt.trip_id.nunique()} trips on route {route_id} "
            f"took {time.time() - route_start:.2f} s"
        )
        print(f"Finished analyzing {ix + 1} of {rt_df.route_id.nunique()} routes")

    # Concatenate all the files previously written for each route. Save them as a new
    # file and delete the smaller files.
    file_date = str(path_to_json).split(".")[0].split("_")[-1]
    all_csvs = list(gtfs_root.glob("realtime_speeds_*.csv"))
    combined_df = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
    all_speeds_file = gtfs_root / f"realtime_link_speeds_{file_date}.csv"
    combined_df.to_csv(all_speeds_file, index=False)
    for f in all_csvs:
        os.remove(f)
    print(
        f"Combined all route CSVs into {all_speeds_file}"
        " and deleted originals."
    )
    print(
        f"Analyzing all {rt_df.trip_id.nunique()} trips "
        f"took {time.time() - first_trip_start:.2f} s"
    )


if __name__ == "__main__":
    json_path = "scripts/gtfs_realtime/greater_portland_me/gtfs_realtime_records_20251023.jsonl"
    get_speeds_for_one_day(json_path)
