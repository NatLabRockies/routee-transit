from functools import partial
import logging
import multiprocessing as mp

import pandas as pd
from gradeit.gradeit import gradeit

from nrel.routee.transit.prediction.grade.tile_resolution import TileResolution
from nrel.routee.transit.prediction.grade.download import CACHE_DIR, download_usgs_tiles

log = logging.getLogger(__file__)


def run_gradeit_parallel(
    trip_dfs_list: list[pd.DataFrame],
    tile_resolution: TileResolution | str = TileResolution.ONE_THIRD_ARC_SECOND,
    n_processes: int = mp.cpu_count(),
) -> pd.DataFrame:
    """Run gradeit in parallel for the provided list of trips.

    Args:
        trip_dfs_list (list[pd.DataFrame]): List of DataFrames, each containing shape
        traces with "shape_pt_lat" and "shape_pt_lon" for a single bus trip.
        raster_path (str | Path): Path to directory holding elevation raster data,
        tile_resolution (TileResolution | str): The resolution of the USGS elevation tiles to use.
            Determines the granularity of elevation data used for grade calculation.
        n_processes (int): Number of processes to run in parallel.

    Returns:
        pd.DataFrame: DataFrame including all input trip/shape data, plus gradeit
            outputs (filtered and unfiltered elevation and grade), for all trips.
    """
    log.info(
        f"Running gradeit on {len(trip_dfs_list)} trips with {n_processes} processes."
    )
    if isinstance(tile_resolution, str):
        tile_resolution = TileResolution.from_string(tile_resolution)

    trip_coords_list = [
        _convert_to_coord_df(trip_link_df) for trip_link_df in trip_dfs_list
    ]
    all_points = [
        (row["lat"], row["lon"])
        for trip_coords_df in trip_coords_list
        for _, row in trip_coords_df.iterrows()
    ]

    download_usgs_tiles(
        lat_lon_pairs=all_points,
        resolution=tile_resolution,
        output_dir=CACHE_DIR,
    )

    gradeit_partial = partial(add_grade_to_trip, tile_resolution=tile_resolution)
    with mp.Pool(n_processes) as pool:
        trips_with_grade = pool.map(gradeit_partial, trip_dfs_list)
    return pd.concat(trips_with_grade)


def _convert_to_coord_df(
    trip_link_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert a trip DataFrame to a coordinate DataFrame for gradeit processing.

    Args:
        trip_link_df (pd.DataFrame): Trip DataFrame where geometry has been aggregated
            by link.

    Returns:
        pd.DataFrame: DataFrame with 'lat' and 'lon' columns for each point in the trip.
    """
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
    return trip_coords_df


def add_grade_to_trip(
    trip_link_df: pd.DataFrame,
    tile_resolution: TileResolution = TileResolution.ONE_THIRD_ARC_SECOND,
) -> pd.DataFrame:
    """Use gradeit to add grade and elevation columns to a trip DataFrame.

    Args:
        trip_link_df (pd.DataFrame): Trip DataFrame where geometry has been aggregated
            by link.
        tile_resolution (TileResolution): The resolution of the USGS tiles to use for
            elevation and grade calculations. Defaults to ONE_ARC_SECOND.

    Returns:
        pd.DataFrame: Trip DataFrame with elevation and grade columns added.
    """
    if isinstance(tile_resolution, str):
        tile_resolution = TileResolution.from_string(tile_resolution)

    # Reset the index to make sure things go as expected when we combine DFs later
    trip_link_df = trip_link_df.reset_index(drop=True)

    # Input DF is link level. Extract the coordinates of all points by grabbing the
    # start coords of each link, then appending the end coords of the last one.
    trip_coords_df = _convert_to_coord_df(trip_link_df)

    # Run gradeit on the coordinate-level DF
    gradeit_out = gradeit(
        df=trip_coords_df,
        source="usgs-local",
        usgs_db_path=CACHE_DIR,
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
