from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nrel.routee.compass.io.generate_dataset import HookParameters

import datetime
import logging
import multiprocessing as mp
from functools import partial

import geopandas as gpd
import pandas as pd
from geopy.distance import great_circle
from gtfsblocks import Feed
from pandas.api.typing import NaTType
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

logger = logging.getLogger("gtfs_processing")

KM_TO_METERS = 1000
FT_TO_METERS = 0.3048
FT_TO_MILES = 0.000189394


def build_corridor_polygon(
    shapes: pd.DataFrame,
    extra_geoms: list[gpd.GeoDataFrame | pd.DataFrame] | None = None,
    buffer_deg: float = 0.01,
    deadhead_buffer_deg: float = 0.05,
) -> BaseGeometry:
    """Build a buffered corridor polygon from GTFS shapes (+ optional extra geometries).

    Each shape line is buffered by ``buffer_deg`` and any deadhead / depot geometries by
    the larger ``deadhead_buffer_deg`` (Compass routes those itself rather than
    map-matching them, so they need more network coverage). Buffering each line then
    unioning is faster than union-then-buffer.

    Used by the OSM build (:meth:`GTFSEnergyPredictor.load_compass_app`, which passes the
    result to ``ox.graph_from_polygon``); exposed as a reusable utility so alternative
    network-download pipelines can cover exactly the same area.
    """
    if shapes.empty:
        raise ValueError("Cannot build a corridor polygon from empty GTFS shapes")

    shape_lines: list[LineString] = []
    for _, grp in shapes.groupby("shape_id"):
        coords = list(zip(grp["shape_pt_lon"], grp["shape_pt_lat"]))
        if len(coords) >= 2:
            shape_lines.append(LineString(coords))

    # Validate each extra geom loudly rather than silently skipping malformed input:
    # a dropped frame would shrink the corridor (a missing depot, uncovered deadheads)
    # and only surface as an unrelated routing/map-matching failure later. Callers
    # build these frames only when they have rows to contribute, so None / non-frame /
    # empty all signal an upstream problem worth failing on.
    extra_points: list[LineString] = []
    for df in extra_geoms or []:
        if df is None or not hasattr(df, "columns"):
            raise TypeError(
                "Each extra_geoms entry must be a (Geo)DataFrame with O-D columns; "
                f"got {type(df).__name__}"
            )
        if (
            "geometry_origin" not in df.columns
            or "geometry_destination" not in df.columns
        ):
            raise ValueError(
                "Each extra_geoms frame must have 'geometry_origin' and "
                "'geometry_destination' columns declaring its O-D endpoints; "
                f"got columns {list(df.columns)}"
            )
        if len(df) == 0:
            raise ValueError(
                "Received an empty extra_geoms frame with no O-D rows; callers should "
                "omit frames that have nothing to contribute rather than passing them."
            )
        for _, row in df.iterrows():
            orig = row["geometry_origin"]
            dest = row["geometry_destination"]
            extra_points.append(LineString([(orig.x, orig.y), (dest.x, dest.y)]))

    buffered = [line.buffer(buffer_deg) for line in shape_lines]
    buffered += [line.buffer(deadhead_buffer_deg) for line in extra_points]
    if not buffered:
        raise ValueError("No usable shape geometries to build a corridor polygon")
    return unary_union(buffered)


def timedelta_to_gtfs_time(td: pd.Timedelta | NaTType) -> str:
    """Convert a timedelta-like value to GTFS HH:MM:SS.

    GTFS times may exceed 24:00:00 for service that runs past midnight.
    NaT values are converted to an empty string.
    """
    if isinstance(td, NaTType):
        return ""
    if not isinstance(td, pd.Timedelta):
        raise TypeError(
            "timedelta_to_gtfs_time only accepts pandas Timedelta or NaTType"
        )
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def write_gtfs_stops(params: "HookParameters", feed: Feed) -> None:
    """
    Hook to write GTFS stop-to-edge mapping after dataset generation.

    Args:
        params: Parameters from generate_compass_dataset
        feed: GTFS feed object
    """
    # 1. Join stops to edges spatially
    edges_gdf = params.edges
    stops_gdf = gpd.GeoDataFrame(
        feed.stops,
        geometry=gpd.points_from_xy(feed.stops.stop_lon, feed.stops.stop_lat),
        crs="EPSG:4326",
    )

    # Find nearest edge for each stop
    stops_projected = stops_gdf.to_crs("EPSG:3857")
    edges_projected = edges_gdf.to_crs("EPSG:3857")

    matched_stops = stops_projected.sjoin_nearest(
        edges_projected[["edge_id", "geometry"]], distance_col="dist"
    )

    # 2. Map stops to trip_id using stop_times
    # stop_times maps trip_id -> stop_id
    # matched_stops maps stop_id -> edge_id
    stop_to_edge = matched_stops.set_index("stop_id")["edge_id"].to_dict()

    stop_edge_records = []
    for _, row in feed.stop_times.iterrows():
        stop_id = row["stop_id"]
        trip_id = row["trip_id"]
        if stop_id in stop_to_edge:
            stop_edge_records.append(
                {"trip_id": trip_id, "edge_id": stop_to_edge[stop_id]}
            )

    stop_edge_df = pd.DataFrame(stop_edge_records).drop_duplicates()

    # 3. Write to CSV
    output_path = params.output_directory / "gtfs_stops.csv"
    stop_edge_df.to_csv(output_path, index=False)
    logger.info(f"Wrote {len(stop_edge_df)} stop-edge mappings to {output_path}")


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
        A single concatenated DataFrame with extended trace information
        for all trips, including estimated timestamps.
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
