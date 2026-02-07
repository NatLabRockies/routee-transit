import datetime
import logging
import multiprocessing as mp
from functools import partial

import geopandas as gpd
import pandas as pd
from geopy.distance import great_circle
from gtfsblocks import Feed

logger = logging.getLogger("gtfs_processing")

KM_TO_METERS = 1000
FT_TO_METERS = 0.3048
FT_TO_MILES = 0.000189394


def upsample_shape(shape_df: pd.DataFrame) -> pd.DataFrame:
    """Upsample a GTFS shape DataFrame to generate a roughly 1 Hz GPS trace.

    Interpolates latitude, longitude, and distance traveled, assuming a constant speed.
    The function performs the following steps:

    * Calculates the distance between consecutive shape points using great-circle distance
    * Computes the cumulative distance traveled along the shape
    * Assigns timestamps based on constant speed (30 km/h)
    * Resamples and interpolates the shape to 1-second intervals
    * Returns DataFrame with interpolated coordinates, timestamps, and distances

    Args:
        shape_df: DataFrame containing GTFS shape points with columns
            'shape_pt_lat', 'shape_pt_lon', and 'shape_id'.

    Returns:
        Upsampled DataFrame with columns 'shape_pt_lat', 'shape_pt_lon',
        'shape_dist_traveled', 'timestamp', and 'shape_id', sampled at 1 Hz.
    """

    # Shift latitude and longitude to get previous point
    shape_df["prev_latitude"] = shape_df["shape_pt_lat"].shift()
    shape_df["prev_longitude"] = shape_df["shape_pt_lon"].shift()

    # Calculate the distance between consecutive points using great_circle
    # TODO: move away from apply() for speed
    shape_df["distance_km"] = shape_df.apply(
        lambda row: (
            great_circle(
                (row["prev_latitude"], row["prev_longitude"]),  # Previous point
                (row["shape_pt_lat"], row["shape_pt_lon"]),  # Current point
            ).kilometers
            if pd.notnull(row["prev_latitude"])
            else 0
        ),
        axis=1,
    )

    # Calculate total distance
    total_distance_km = shape_df["distance_km"].sum()

    # Use calculated total distance instead of shape_dist_traveled
    shape_df["shape_dist_traveled"] = shape_df["distance_km"].cumsum()

    # Speed is assumed to be 30 km/h, which is about 10 (8.33) m per second/node
    shape_df["segment_duration_delta"] = (
        shape_df["shape_dist_traveled"]
        / shape_df["shape_dist_traveled"].max()
        * datetime.timedelta(seconds=round(total_distance_km / 30 * 3600))
    )
    shape_df["segment_duration_delta"] = shape_df["segment_duration_delta"].apply(
        lambda x: datetime.timedelta(seconds=round(x.total_seconds()))
    )
    # Define an arbitrary date to convert from timedelta to datetime
    date_tmp = pd.Timestamp(datetime.datetime(2023, 9, 3))
    shape_df["timestamp"] = date_tmp + shape_df["segment_duration_delta"]

    # Upsample to 1s
    shape_id_tmp = shape_df.shape_id.iloc[0]
    shape_df = (
        shape_df[["shape_pt_lat", "shape_pt_lon", "timestamp", "shape_dist_traveled"]]
        .drop_duplicates(subset=["timestamp"])
        .set_index("timestamp")
        .resample("1s")
        .interpolate(method="linear")
    )

    # Now we have the 1 Hz gps trace for each trip with timestamp
    shape_df = shape_df.reset_index(drop=True)
    shape_df["shape_id"] = shape_id_tmp

    return shape_df


def add_stop_flags_to_shape(
    trip_shape_df: pd.DataFrame, stop_times_ext: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Attach stop information to a DataFrame of shape points for a specific trip.

    Given a DataFrame of shape points (`trip_shape_df`) and a DataFrame of stop times
    (`stop_times_ext`) joined with stop locations, this function identifies which shape
    points correspond to stops for the trip and annotates them.

    Parameters
    ----------
    trip_shape_df : pd.DataFrame
        DataFrame containing shape points for a single trip. Must include columns
        'trip_id', 'shape_pt_lon', 'shape_pt_lat', and 'coordinate_id'.
    stop_times_ext : pd.DataFrame
        DataFrame containing stop times with extended information. Must include columns
        'trip_id', 'stop_lon', and 'stop_lat'.
    Returns
    -------
    pd.DataFrame
        The input DataFrame with an additional column 'with_stop', where 1 indicates
        the shape point is nearest to a stop, and 0 otherwise.
    Notes
    -----
    - Uses spatial join to find the nearest shape point for each stop.
    """
    # Confirm we're only getting a single trip id
    if trip_shape_df["trip_id"].nunique() > 1:
        raise ValueError(
            f"trip_shape_df should only contain data for a single trip, but the "
            f"input includes {trip_shape_df.trip_id.nunique()} different trip IDs."
        )

    # Filter down stop_times to only the given trip from
    trip_id = trip_shape_df["trip_id"].iloc[0]
    trip_stop_times = stop_times_ext[stop_times_ext.trip_id == trip_id]

    # Convert DFs to GeoDataFrame for spatial join
    trip_gdf = gpd.GeoDataFrame(
        trip_shape_df,
        geometry=gpd.points_from_xy(
            trip_shape_df.shape_pt_lon, trip_shape_df.shape_pt_lat
        ),
    )
    stop_times_gdf = gpd.GeoDataFrame(
        trip_stop_times,
        geometry=gpd.points_from_xy(trip_stop_times.stop_lon, trip_stop_times.stop_lat),
    )

    # TODO: handle downstream effects of this warning, or change to an error
    if (~trip_gdf.geometry.is_valid).any():
        logger.warning(f"Invalid geometry detected for trip {trip_id}")

    stop_times_gdf = stop_times_gdf.sjoin_nearest(
        trip_gdf[["geometry", "coordinate_id"]]
    )
    trip_gdf["with_stop"] = 0
    trip_gdf.loc[
        trip_gdf.coordinate_id.isin(stop_times_gdf.coordinate_id.to_list()), "with_stop"
    ] = 1

    df_tmp = trip_gdf.drop(["geometry"], axis=1)
    return df_tmp


def estimate_trip_timestamps(trip_shape_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate timestamps for each shape point of a trip based on distance traveled.

    Args:
        trip_shape_df (pd.DataFrame): DataFrame containing trip shape data with columns:
            - 'shape_dist_traveled': Cumulative distance traveled along the shape.
            - 'start_time': Origin time (datetime) of the trip.
            - 'end_time': Destination time (datetime) of the trip.
    Returns:
        pd.DataFrame: Modified DataFrame with additional columns:
            - 'segment_duration_delta': Estimated duration for each segment as timedelta.
            - 'timestamp': Estimated timestamp for each segment.
            - 'Datetime_nearest5': Timestamp rounded to the nearest 5 minutes.
            - 'hour': Hour component of the rounded timestamp.
            - 'minute': Minute component of the rounded timestamp.
    """
    start_times = pd.to_timedelta(trip_shape_df["start_time"])
    end_times = pd.to_timedelta(trip_shape_df["end_time"])
    trip_shape_df["segment_duration_delta"] = (
        trip_shape_df["shape_dist_traveled"]
        / (trip_shape_df["shape_dist_traveled"].max() + 0.0001)
        * (end_times - start_times)
    )
    trip_shape_df["segment_duration_delta"] = trip_shape_df[
        "segment_duration_delta"
    ].apply(lambda x: datetime.timedelta(seconds=round(x.total_seconds())))
    trip_shape_df["timestamp"] = start_times + trip_shape_df["segment_duration_delta"]

    ## get hour and minute of gps timestamp
    trip_shape_df["Datetime_nearest5"] = trip_shape_df["timestamp"].dt.round("5min")
    trip_shape_df["hour"] = trip_shape_df["Datetime_nearest5"].dt.components["hours"]
    trip_shape_df["minute"] = trip_shape_df["Datetime_nearest5"].dt.components[
        "minutes"
    ]

    return trip_shape_df


def extend_trip_traces(
    trips_df: pd.DataFrame,
    matched_shapes_df: pd.DataFrame,
    feed: Feed,
    add_stop_flag: bool = False,
    n_processes: int | None = mp.cpu_count(),
) -> pd.DataFrame:
    """Extend trip shapes with stop details and estimated timestamps from GTFS.

    This function processes GTFS trip and shape data to:

    * Summarize stop times for each trip (first/last stop and times)
    * Merge stop time summaries into the trips DataFrame
    * Attach stop coordinates to stop times
    * Merge trip and shape data to create ordered trip traces
    * Optionally, attach stop indicators to shape trace points
    * Estimate timestamps for each trace point based on scheduled trip duration and distance

    Args:
        trips_df: DataFrame containing trip information, including
            'trip_id' and 'shape_id'.
        matched_shapes_df: DataFrame with shape points matched to trips,
            including 'shape_id' and 'shape_dist_traveled'.
        feed: GTFS feed object containing 'stop_times' and 'stops'
            DataFrames.
        add_stop_flag: If True, attaches stop indicators to shape trace
            points. Defaults to False.
        n_processes: Number of processes to run in parallel using
            multiprocessing. Defaults to mp.cpu_count().

    Returns:
        A list of DataFrames, one per trip, with extended trace information
        including estimated timestamps.
    """
    # Add stop coordinates to stop_times
    stop_times_ext = feed.stop_times[["trip_id", "stop_sequence", "stop_id"]].merge(
        feed.stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id"
    )

    # Calculate approximate timestamps for each trip
    trip_shape = pd.merge(
        trips_df[["trip_id", "shape_id", "start_time", "end_time"]],
        matched_shapes_df,
        how="left",
        on="shape_id",
    )
    trip_shape = trip_shape.sort_values(
        by=["trip_id", "shape_dist_traveled"]
    ).reset_index(drop=True)
    trip_shapes_list = [item for _, item in trip_shape.groupby("trip_id")]

    # Attach stops to shape traces. Note that this just adds a dummy variable column
    # indicating whether or not a stop is located at a given point on the shape.
    if add_stop_flag:
        attach_stop_partial = partial(
            add_stop_flags_to_shape, stop_times_ext=stop_times_ext
        )
        with mp.Pool(n_processes) as pool:
            trip_shapes_list = pool.map(attach_stop_partial, trip_shapes_list)

    # Attach timestamps to each trip. These are simply based on the scheduled trip
    # duration and shape_dist_traveled, assuming a constant speed for the entire trip.
    # TODO: improve timestamp estimates
    with mp.Pool(n_processes) as pool:
        trips_with_timestamps_list = pool.map(
            estimate_trip_timestamps, trip_shapes_list
        )
    logger.info("Finished attaching timestamps")
    return pd.concat(trips_with_timestamps_list)
