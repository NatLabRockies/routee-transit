import logging
from typing import Any

import geopandas as gpd
import pandas as pd
import shapely

from nrel.routee.compass import CompassApp
from nrel.routee.compass.utils.geometry import geometry_from_route

log = logging.getLogger(__name__)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Get the haversine distance between two points in kilometers."""
    import math

    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def route_single_trip_fallback(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    block_id: str,
) -> list[dict[str, Any]]:
    """
    Return a simple straight-line link from start to end if no route found.
    """
    return [
        {
            "shape_id": block_id,
            "shape_pt_sequence": 1,
            "shape_pt_lon": float(start_lon),
            "shape_pt_lat": float(start_lat),
            "shape_dist_traveled": 0.0,
        },
        {
            "shape_id": block_id,
            "shape_pt_sequence": 2,
            "shape_pt_lon": float(end_lon),
            "shape_pt_lat": float(end_lat),
            "shape_dist_traveled": _haversine_km(
                start_lat, start_lon, end_lat, end_lon
            ),
        },
    ]


def create_deadhead_shapes(
    app: CompassApp,
    df: gpd.GeoDataFrame,
    o_col: str = "geometry_origin",
    d_col: str = "geometry_destination",
) -> pd.DataFrame:
    """
    Compute deadhead route shapes between origin and destination.

    For each row in `df`, this computes a shortest-path using CompassApp.
    Returns a pandas DataFrame with per-point GTFS-like shape rows.

    Parameters
    ----------
    app : CompassApp
        The CompassApp instance to use for routing.
    df : gpd.GeoDataFrame
        DataFrame with origin and destination geometry columns
    o_col : str, optional
        Column name for origin geometries (default: "geometry_origin")
    d_col : str, optional
        Column name for destination geometries (default: "geometry_destination")

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['shape_id', 'shape_pt_sequence', 'shape_pt_lon',
        'shape_pt_lat', 'shape_dist_traveled'] where `shape_dist_traveled` is
        cumulative distance in kilometers from the route start.
    """
    # Prepare queries
    queries = []
    df_indices = []
    for i, r in df.iterrows():
        origin = r[o_col]
        destination = r[d_col]
        queries.append(
            {
                "origin_x": float(origin.x),
                "origin_y": float(origin.y),
                "destination_x": float(destination.x),
                "destination_y": float(destination.y),
                "model_name": "Transit_Bus_Battery_Electric",
                "weights": {"trip_time": 1.0},
            }
        )
        df_indices.append(i)

    # Run queries in parallel
    results = app.run(queries)
    if isinstance(results, dict):
        results = [results]

    # Flatten results into GTFS shape point rows
    shape_rows = []
    for i, result in zip(df_indices, results):
        block_id = df.loc[i].get("block_id")
        origin = df.loc[i][o_col]
        destination = df.loc[i][d_col]

        if "error" in result or result.get("route") is None:
            # Fallback to straight line if error or no route
            cp_error = result.get("error", "No route found")
            log.warning(
                f"CompassApp failed for block_id {block_id}: {cp_error}. "
                "Creating a straight-line fallback route."
            )
            rows = route_single_trip_fallback(
                origin.x, origin.y, destination.x, destination.y, block_id
            )
            shape_rows.extend(rows)
            continue

        line = geometry_from_route(result["route"])

        if line.is_empty:
            log.warning(
                f"CompassApp returned an empty route for block_id {block_id}. "
                "Creating a straight-line fallback route."
            )
            rows = route_single_trip_fallback(
                origin.x, origin.y, destination.x, destination.y, block_id
            )
            shape_rows.extend(rows)
            continue

        coords = shapely.get_coordinates(line)
        prev_lat, prev_lon = None, None
        cum_km = 0.0
        for seq, (lon, lat) in enumerate(coords, start=1):
            if prev_lat is not None and prev_lon is not None:
                cum_km += _haversine_km(prev_lat, prev_lon, lat, lon)
            shape_rows.append(
                {
                    "shape_id": block_id,
                    "shape_pt_sequence": int(seq),
                    "shape_pt_lon": float(lon),
                    "shape_pt_lat": float(lat),
                    "shape_dist_traveled": float(cum_km),
                }
            )
            prev_lat, prev_lon = lat, lon

    out_df = pd.DataFrame(
        shape_rows,
        columns=[
            "shape_id",
            "shape_pt_sequence",
            "shape_pt_lon",
            "shape_pt_lat",
            "shape_dist_traveled",
        ],
    )

    return out_df
