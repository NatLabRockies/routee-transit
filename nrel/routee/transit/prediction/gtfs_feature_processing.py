import datetime
import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path

import geopandas as gpd
import pandas as pd
from geopy.distance import great_circle
from gradeit.gradeit import gradeit
from gtfsblocks import Feed
from mappymatch.constructs.geofence import Geofence
from mappymatch.constructs.trace import Trace
from mappymatch.maps.nx.nx_map import NetworkType, NxMap
from mappymatch.matchers.lcss.lcss import LCSSMatcher
from numpy.random import default_rng

logger = logging.getLogger("gtfs_feature_processing")

KM_TO_METERS = 1000
FT_TO_METERS = 0.3048
FT_TO_MILES = 0.000189394


def upsample_shape(shape_df: pd.DataFrame) -> pd.DataFrame:
    """Upsample a GTFS shape DataFrame to generate a roughly 1 Hz GPS trace.

    Interpolates latitude, longitude, and distance traveled, assuming a constant speed.
    The function performs the following steps:
    - Calculates the distance between consecutive shape points using the great-circle
        distance.
    - Computes the cumulative distance traveled along the shape.
    - Assigns timestamps to each point based on a constant speed (30 km/h).
    - Resamples and linearly interpolates the shape to 1-second intervals.
    - Returns a DataFrame with interpolated latitude, longitude, timestamp, distance
        traveled, and shape ID.

    Args:
        shape_df (pd.DataFrame): DataFrame containing GTFS shape points with columns
            'shape_pt_lat', 'shape_pt_lon', and 'shape_id'.
    Returns:
        pd.DataFrame: Upsampled DataFrame with columns 'shape_pt_lat', 'shape_pt_lon',
        'shape_dist_traveled', 'timestamp', and 'shape_id', sampled at 1 Hz.
    """

    # Shift latitude and longitude to get previous point
    shape_df["prev_latitude"] = shape_df["shape_pt_lat"].shift()
    shape_df["prev_longitude"] = shape_df["shape_pt_lon"].shift()

    # Calculate the distance between consecutive points using great_circle
    # TODO: move away from apply() for speed
    shape_df["distance_km"] = shape_df.apply(
        lambda row: great_circle(
            (row["prev_latitude"], row["prev_longitude"]),  # Previous point
            (row["shape_pt_lat"], row["shape_pt_lon"]),  # Current point
        ).kilometers
        if pd.notnull(row["prev_latitude"])
        else 0,
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
    date_tmp = datetime.datetime(2023, 9, 3)
    shape_df["timestamp"] = (
        datetime.timedelta(seconds=0) + shape_df["segment_duration_delta"] + date_tmp
    )

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


def match_shape_to_osm(upsampled_shape_df: pd.DataFrame) -> pd.DataFrame:
    """Match a given GTFS shape DataFrame to the OpenStreetMap (OSM) road network.

    This function uses mappymatch to add OSM network information to the shape trace.
    The trace should be upsampled beforehand to approximately 1 Hz/8 m for the most
    accurate expected mapping performance. The function creates a Trace from the input
    DataFrame, constructs a geofence around the trace, extracts the OSM road network
    within the geofence, and applies the mappymatch LCSS matcher to align the trace to
    the network. The output DataFrame retains the full shape while adding network
    information to each row.

    Args:
        upsampled_shape_df (pd.DataFrame): DataFrame containing the shape points with
            latitude and longitude columns ("shape_pt_lat" and "shape_pt_lon").
    Returns:
        pd.DataFrame: A DataFrame combining the original upsampled shape points with
            their corresponding OSM network matches.
    """
    # Create mappymatch trace
    trace = Trace.from_dataframe(
        upsampled_shape_df, lat_column="shape_pt_lat", lon_column="shape_pt_lon"
    )
    # Create geofence and use it to pull network
    geofence = Geofence.from_trace(trace, padding=1e3)
    nxmap = NxMap.from_geofence(geofence, network_type=NetworkType.DRIVE)
    # Run map matching algorithm
    matcher = LCSSMatcher(nxmap)
    matches = matcher.match_trace(trace).matches_to_dataframe()
    # Combine shape with network details
    df_result = pd.concat([upsampled_shape_df, matches], axis=1)
    return df_result


def estimate_trip_timestamps(trip_shape_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate timestamps for each shape point of a trip based on distance traveled.

    Args:
        trip_shape_df (pd.DataFrame): DataFrame containing trip shape data with columns:
            - 'shape_dist_traveled': Cumulative distance traveled along the shape.
            - 'o_time': Origin time (datetime) of the trip.
            - 'd_time': Destination time (datetime) of the trip.
    Returns:
        pd.DataFrame: Modified DataFrame with additional columns:
            - 'segment_duration_delta': Estimated duration for each segment as timedelta.
            - 'timestamp': Estimated timestamp for each segment.
            - 'Datetime_nearest5': Timestamp rounded to the nearest 5 minutes.
            - 'hour': Hour component of the rounded timestamp.
            - 'minute': Minute component of the rounded timestamp.
    """
    trip_shape_df["segment_duration_delta"] = (
        trip_shape_df["shape_dist_traveled"]
        / trip_shape_df["shape_dist_traveled"].max()
        * (trip_shape_df["d_time"] - trip_shape_df["o_time"])
    )
    trip_shape_df["segment_duration_delta"] = trip_shape_df[
        "segment_duration_delta"
    ].apply(lambda x: datetime.timedelta(seconds=round(x.total_seconds())))
    trip_shape_df["timestamp"] = (
        trip_shape_df["o_time"] + trip_shape_df["segment_duration_delta"]
    )

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
) -> list[pd.DataFrame]:
    """Extend trip shapes with stop details and estimated timestamps from GTFS.

    This function processes GTFS trip and shape data to:
    - Summarize stop times for each trip (first/last stop and times).
    - Merge stop time summaries into the trips DataFrame.
    - Attach stop coordinates to stop times.
    - Merge trip and shape data to create ordered trip traces.
    - Optionally, attach stop indicators to shape trace points.
    - Estimate timestamps for each trace point based on scheduled trip duration and
      distance.

    Args:
        trips_df (pd.DataFrame): DataFrame containing trip information, including
            'trip_id' and 'shape_id'.
        matched_shapes_df (pd.DataFrame): DataFrame with shape points matched to trips,
            including 'shape_id' and 'shape_dist_traveled'.
        feed (gtfsblocks.Feed): GTFS feed object containing 'stop_times' and 'stops'
            DataFrames.
        add_stop_flag (bool, optional): If True, attaches stop indicators to shape trace
            points. Defaults to False.
        n_processes (int | None, optional): Number of processes to run in parallel using
            multiprocessing. Defaults to mp.cpu_count().

    Returns:
        list: A list of DataFrames, one per trip, with extended trace information
            including estimated timestamps.
    """

    # 3) Estimate speeds
    # Start by summarizing stop times: get first and last stop, plus start/end times
    stop_times_by_trip = (
        feed.stop_times.groupby("trip_id")
        .agg(
            {
                "arrival_time": "first",
                "departure_time": "last",
                "stop_id": ["first", "last"],
            }
        )
        .reset_index()
    )
    stop_times_by_trip.columns = [
        "trip_id",
        "o_time",
        "d_time",
        "o_stop_id",
        "d_stop_id",
    ]

    # Add start/end times and stops to trips DF
    # TODO: consider doing this with gtfsblocks add_trip_data()
    trips_df = pd.merge(trips_df, stop_times_by_trip, how="left", on="trip_id")
    trips_df["o_time"] = pd.to_timedelta(trips_df["o_time"])
    trips_df["d_time"] = pd.to_timedelta(trips_df["d_time"])
    trips_df["trip_duration"] = trips_df["d_time"] - trips_df["o_time"]

    # Add stop coordinates to stop_times
    stop_times_ext = feed.stop_times[["trip_id", "stop_sequence", "stop_id"]].merge(
        feed.stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id"
    )

    # calculate approximate timestamps for each GPS trace
    # TODO: I think this big merge can be avoided
    trip_shape = pd.merge(
        trips_df[["trip_id", "shape_id", "o_time", "d_time"]],
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
    return trips_with_timestamps_list


def run_gradeit_parallel(
    trip_dfs_list: list[pd.DataFrame],
    raster_path: str | Path,
    n_processes: int,
) -> pd.DataFrame:
    """Run gradeit in parallel for the provided list of trips.

    Args:
        trip_dfs_list (list[pd.DataFrame]): List of DataFrames, each containing shape
        traces with "shape_pt_lat" and "shape_pt_lon" for a single bus trip.
        raster_path (str | Path): Path to directory holding elevation raster data,
            supplied as `usgs_db_path` to gradeit.
        n_processes (int): Number of processes to run in parallel.

    Returns:
        pd.DataFrame: DataFrame including all input trip/shape data, plus gradeit
            outputs (filtered and unfiltered elevation and grade), for all trips.
    """
    gradeit_partial = partial(add_grade_to_trip, raster_path=raster_path)
    with mp.Pool(n_processes) as pool:
        trips_with_grade = pool.map(gradeit_partial, trip_dfs_list)
    return pd.concat(trips_with_grade)


def add_grade_to_trip(
    trip_link_df: pd.DataFrame, raster_path: str | Path
) -> pd.DataFrame:
    """Use gradeit to add grade and elevation columns to a trip DataFrame.

    Args:
        trip_link_df (pd.DataFrame): Trip DataFrame where geometry has been aggregated
            by link.
        raster_path (str | Path): Path to USGS elevation tiles.

    Returns:
        pd.DataFrame: Trip DataFrame with elevation and grade columns added.
    """
    # Reset the index to make sure things go as expected when we combine DFs later
    trip_link_df = trip_link_df.reset_index(drop=True)

    # Input DF is link level. Extract the coordinates of all points by grabbing the
    # start coords of each link, then appending the end coords of the last one.
    trip_coords_df = trip_link_df[["start_lat", "start_lon"]].rename(
        columns={"start_lat": "lat", "start_lon": "lon"}
    )
    last_rw = trip_link_df.iloc[-1][["end_lat", "end_lon"]].T
    trip_coords_df = pd.concat(
        [
            trip_coords_df,
            last_rw.to_frame().rename({"end_lat": "lat", "end_lon": "lon"}).T,
        ],
        ignore_index=True,
    )

    # Run gradeit on the coordinate-level DF
    gradeit_out = gradeit(
        df=trip_coords_df,
        source="usgs-local",
        usgs_db_path=raster_path,
        lat_col="lat",
        lon_col="lon",
        filtering=True,
    )

    # When calculating grades, the first point from gradeit has zero distance/grade.
    # We'll shift these to appropriately match to links.
    gradeit_cols = [
        "elevation_ft",
        "distances_ft",
        "grade_dec_unfiltered",
        "elevation_ft_filtered",
        "grade_dec_filtered",
    ]
    trip_link_df.loc[:, gradeit_cols] = gradeit_out.shift(-1)[:-1].loc[:, gradeit_cols]

    return trip_link_df


def build_routee_features_with_osm(
    agency: str,
    n_trips: int | None = 100,
    add_road_grade: bool = False,
    gradeit_tile_path: Path | str | None = None,
    n_processes: int = mp.cpu_count(),
) -> pd.DataFrame:
    """Process a GTFS feed to provide inputs for RouteE-powertrain energy prediction.

    This wrapper function processes a GTFS feed to estimate link-level bus speeds and
    (optionally) elevation change for all scheduled trips. Optionally, n_trips can be
    used to select only a random subset for faster testing and validation.

    This function reads the GTFS data into a gtfsblocks.Feed object, matches all
    relevant trip shapes to the OpenStreetMap network using mappymatch, and optionally
    adds road grade information using gradeit. The output DataFrame includes the
    features needed to run energy consumption prediction with a RouteE vehicle model.

    Args:
        agency (str): Name of the transit agency under study. This must correspond to
            the name of a directory under `data/gtfs` containing a fully populated
            GTFS feed.
        n_trips (int | None, optional): The number of trips to include in the analysis.
            If None, all trips will be included. If an integer, that number of trips
            will be selected at random. Defaults to 100.
        add_road_grade (bool, optional): Whether to append road grade information.
            Requires local elevation raster files specified with `gradeit_tile_path`.
            Defaults to False.
        gradeit_tile_path (Path | str | None, optional): Path to a directory containing.
            elevation tiles to be used by `gradeit`. Defaults to None.
        n_processes (int | None, optional): Number of processes to run in parallel using
            multiprocessing. Defaults to mp.cpu_count().

    Raises:
        ValueError: If `gradeit_tile_path` is None and `add_road_grade` is True.

    Returns:
        pd.DataFrame: DataFrame with link-level speed, distance, and (optionally)
            grade for all bus trips in scope.
    """
    # 1) Process GTFS inputs
    req_cols = {
        "stop_times": [
            "arrival_time",
            "departure_time",
            "shape_dist_traveled",
            "stop_id",
        ],
        "shapes": ["shape_dist_traveled"],
    }
    gtfs_path = f"data/gtfs/{agency}"
    feed = Feed.from_dir(gtfs_path, columns=req_cols)
    logger.info(f"Feed contains {len(feed.trips)} trips and {len(feed.shapes)} shapes")

    # 1.5) Filter down feed to speed up testing
    if n_trips is not None:
        rng = default_rng(seed=100)
        trips_incl = rng.choice(feed.trips.trip_id.unique(), n_trips, replace=False)
        trips_df = feed.trips[feed.trips.trip_id.isin(trips_incl)]
        shapes_incl = trips_df.shape_id.unique()
        shapes_df = feed.shapes[feed.shapes.shape_id.isin(shapes_incl)]
        logger.info(
            f"Restricted feed to {len(trips_df)} trips and {len(shapes_incl)} shapes"
        )
    else:
        trips_df = feed.trips
        shapes_df = feed.shapes

    # 2) Refine shapes
    # Upsample all shapes
    df_shape_list = [group for _, group in shapes_df.groupby("shape_id")]
    with mp.Pool(n_processes) as pool:
        upsampled_shapes_list = pool.map(upsample_shape, df_shape_list)
    logger.info("Finished upsampling")
    logger.info("Original shapes length: {}".format(len(shapes_df)))
    logger.info(
        "Upsampled shapes length: {}".format(len(pd.concat(upsampled_shapes_list)))
    )

    # Run mapmatching in parallel for each shape
    with mp.Pool(n_processes) as pool:
        matched_shapes_list = pool.map(match_shape_to_osm, upsampled_shapes_list)

    logger.info("Finished map matching")

    # matched_shapes_df is a large dataframe in which each row is a location of the
    # upsampled shape and corresponding map data.
    matched_shapes_df = pd.concat(matched_shapes_list)

    # Extend trip data with stop and schedule data
    trips_ext_list = extend_trip_traces(
        trips_df=trips_df,
        matched_shapes_df=matched_shapes_df,
        feed=feed,
        add_stop_flag=False,
    )
    trips_df_ext = pd.concat(trips_ext_list)

    # Aggregate data at road link level to reduce computational burden
    trip_links_df = (
        trips_df_ext.groupby(by=["trip_id", "shape_id", "road_id"])
        .agg(
            start_lat=pd.NamedAgg("shape_pt_lat", "first"),
            start_lon=pd.NamedAgg("shape_pt_lon", "first"),
            end_lat=pd.NamedAgg("shape_pt_lat", "last"),
            end_lon=pd.NamedAgg("shape_pt_lon", "last"),
            geom=pd.NamedAgg("geom", "first"),
            start_timestamp=pd.NamedAgg("timestamp", "first"),
            end_timestamp=pd.NamedAgg("timestamp", "last"),
            kilometers=pd.NamedAgg("kilometers", "mean"),
            travel_time_osm=pd.NamedAgg("travel_time", "mean"),
        )
        .reset_index()
    )
    trips_df_list = [t_df for _, t_df in trip_links_df.groupby("trip_id")]

    if add_road_grade:
        if gradeit_tile_path is None:
            raise ValueError(
                "A path to map tiles must be passed to build_routee_features_with_osm()"
                "in order to add road grade."
            )

        result_df = run_gradeit_parallel(
            trip_dfs_list=trips_df_list,
            raster_path=gradeit_tile_path,
            n_processes=n_processes,
        )
    else:
        result_df = pd.concat(trips_df_list)

    return result_df
