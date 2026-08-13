import logging
from typing import Any

import geopandas as gpd
import pandas as pd
import shapely
from nrel.routee.compass.utils.geometry import geometry_from_route

from .compass_app import TransitCompassApp

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


_WEEKDAYS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


def gtfs_time_to_query_time(
    departure: Any, base_weekday: str | None
) -> tuple[str, str] | None:
    """
    Convert a GTFS departure time into a Compass ``(start_time, start_weekday)`` pair.

    GTFS times are measured as an offset from the service day's midnight and may
    exceed ``24:00:00`` for service running past midnight. A
    ``speed_time_of_day`` traversal model parses ``start_time`` with chrono's
    ``"%H:%M:%S"`` (hours 0-23 only), so past-midnight offsets are wrapped into the
    0-23h range and the weekday is rolled forward by the number of whole days.

    Parameters
    ----------
    departure :
        A pandas Timedelta / timedelta-like (offset from service-day midnight).
        ``NaT``/``None``/negative values yield ``None``.
    base_weekday :
        Lower-case weekday name for the service day (e.g. ``"wednesday"``). When
        ``None`` or unrecognized, defaults to ``"monday"``.

    Returns
    -------
    tuple[str, str] | None
        ``(start_time "HH:MM:SS", start_weekday)`` or ``None`` when no usable time.
    """
    if departure is None:
        return None
    td = pd.to_timedelta(departure, errors="coerce")
    if td is pd.NaT or pd.isna(td):
        return None
    total_seconds = int(td.total_seconds())
    if total_seconds < 0:
        return None

    days, remainder = divmod(total_seconds, 86400)
    hours = remainder // 3600
    minutes = (remainder % 3600) // 60
    seconds = remainder % 60
    start_time = f"{hours:02}:{minutes:02}:{seconds:02}"

    base = (base_weekday or "monday").lower()
    if base not in _WEEKDAYS:
        base = "monday"
    weekday = _WEEKDAYS[(_WEEKDAYS.index(base) + days) % 7]
    return start_time, weekday


def _create_od_key(
    origin_x: float, origin_y: float, dest_x: float, dest_y: float, precision: int = 3
) -> str:
    """
    Create a deterministic key for an origin-destination pair.

    Uses rounded coordinates to identify "equivalent" O-D pairs.
    Default precision of 3 decimal places corresponds to ~100m accuracy.

    Parameters
    ----------
    origin_x, origin_y : float
        Origin coordinates (longitude, latitude)
    dest_x, dest_y : float
        Destination coordinates (longitude, latitude)
    precision : int, optional
        Number of decimal places for coordinate rounding (default: 3)

    Returns
    -------
    str
        Unique key identifying this O-D pair
    """
    ox = round(origin_x, precision)
    oy = round(origin_y, precision)
    dx = round(dest_x, precision)
    dy = round(dest_y, precision)
    return f"{ox},{oy}->{dx},{dy}"


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
    app: TransitCompassApp,
    df: gpd.GeoDataFrame,
    o_col: str = "geometry_origin",
    d_col: str = "geometry_destination",
    min_distance_m: float = 200.0,
    start_weekday: str | None = None,
    time_col: str = "departure_time",
    model_name: str = "Transit_Bus_Battery_Electric",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute deadhead route shapes for unique origin-destination pairs.

    This function identifies unique O-D pairs from the input DataFrame and routes
    only those pairs, significantly reducing the routing burden when many trips
    share the same O-D pair.

    Parameters
    ----------
    app : TransitCompassApp
        The TransitCompassApp instance to use for routing.
    df : gpd.GeoDataFrame
        DataFrame with origin and destination geometry columns. Must include
        a 'block_id' column to identify each input row.
    o_col : str, optional
        Column name for origin geometries (default: "geometry_origin")
    d_col : str, optional
        Column name for destination geometries (default: "geometry_destination")
    min_distance_m : float, optional
        Minimum distance in meters between O and D to perform routing.
        O-D pairs closer than this use straight-line fallback. (default: 200.0)
    start_weekday : str, optional
        Lower-case weekday name for the service day (e.g. ``"wednesday"``), used
        to populate the ``start_weekday`` field of each routing query. Defaults to
        ``"monday"`` when not provided. Required (together with a per-row
        departure time) by time-of-day traversal models such as the
        ``speed_time_of_day`` model, which reject queries missing ``start_time``.
    time_col : str, optional
        Name of the column in ``df`` holding each row's GTFS departure time
        (offset from the service day's midnight). When present, the departure time
        of the first trip seen for each unique O-D pair is attached to that pair's
        query as ``start_time``/``start_weekday``. (default: ``"departure_time"``)
    model_name : str, optional
        Vehicle model name used to build the routing traversal model. Deadhead
        routing only optimizes ``trip_time``, so this only needs to be a model
        that is loaded in ``app``; it does not affect the resulting geometry.
        (default: ``"Transit_Bus_Battery_Electric"``)

    Notes
    -----
    O-D pairs are de-duplicated before routing, so when several trips share an O-D
    pair the representative departure time of the first such trip is used for the
    shared shape. This is sufficient for shape geometry (which is essentially
    time-independent); time-of-day energy is computed later from the actual GTFS
    timestamps sampled along the resulting shape.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple of (shapes_df, od_mapping_df):

        - shapes_df: DataFrame with columns ['shape_id', 'shape_pt_sequence',
          'shape_pt_lon', 'shape_pt_lat', 'shape_dist_traveled']
        - od_mapping_df: DataFrame with columns ['block_id', 'od_key', 'shape_id']
          mapping each input block_id to its assigned shape
    """
    # Step 1: Compute od_key and distance for each row
    od_data = []
    for i, r in df.iterrows():
        origin = r[o_col]
        destination = r[d_col]
        od_key = _create_od_key(origin.x, origin.y, destination.x, destination.y)
        distance_km = _haversine_km(origin.y, origin.x, destination.y, destination.x)
        od_data.append(
            {
                "block_id": r.get("block_id"),
                "od_key": od_key,
                "origin_x": float(origin.x),
                "origin_y": float(origin.y),
                "dest_x": float(destination.x),
                "dest_y": float(destination.y),
                "distance_m": distance_km * 1000,
                # Representative GTFS departure time for this O-D pair, used to
                # populate start_time on time-of-day routing queries (NaT when the
                # input lacks the column).
                "departure_time": r.get(time_col, pd.NaT),
            }
        )

    od_df = pd.DataFrame(od_data)

    # Step 2: Identify unique O-D pairs
    unique_ods = od_df.drop_duplicates(subset=["od_key"]).copy()
    log.info(
        f"Deadhead routing: {len(od_df)} trips reduced to {len(unique_ods)} unique O-D pairs"
    )

    # Step 3: Separate pairs by distance threshold
    close_ods = unique_ods[unique_ods["distance_m"] < min_distance_m]
    far_ods = unique_ods[unique_ods["distance_m"] >= min_distance_m]

    shape_rows: list[dict[str, Any]] = []

    # Step 4: Create straight-line fallback for close O-D pairs
    for _, row in close_ods.iterrows():
        od_key = row["od_key"]
        rows = route_single_trip_fallback(
            row["origin_x"],
            row["origin_y"],
            row["dest_x"],
            row["dest_y"],
            od_key,  # Use od_key as shape_id
        )
        shape_rows.extend(rows)

    # Step 5: Route far O-D pairs with CompassApp
    if len(far_ods) > 0:
        queries = []
        for _, row in far_ods.iterrows():
            query: dict[str, Any] = {
                "origin_x": row["origin_x"],
                "origin_y": row["origin_y"],
                "destination_x": row["dest_x"],
                "destination_y": row["dest_y"],
                "model_name": model_name,
                "weights": {"trip_time": 1.0},
            }
            # Time-of-day traversal models (e.g. speed_time_of_day) require
            # a start_time/start_weekday on every query; supply them from the
            # GTFS departure time when available, falling back to midday so the
            # query still builds when a trip's departure time is missing.
            time_pair = gtfs_time_to_query_time(
                row.get("departure_time"), start_weekday
            ) or ("12:00:00", (start_weekday or "monday"))
            query["start_time"], query["start_weekday"] = time_pair
            queries.append(query)

        far_rows_by_key: dict[str, dict[str, float]] = {
            str(row["od_key"]): {
                "origin_x": float(row["origin_x"]),
                "origin_y": float(row["origin_y"]),
                "dest_x": float(row["dest_x"]),
                "dest_y": float(row["dest_y"]),
            }
            for _, row in far_ods.iterrows()
        }

        # Run queries in parallel
        results = app.run(queries)
        if isinstance(results, dict):
            results = [results]

        processed_keys: set[str] = set()

        # Process routing results
        for result in results:
            req = result.get("request", {})
            try:
                od_key = _create_od_key(
                    float(req["origin_x"]),
                    float(req["origin_y"]),
                    float(req["destination_x"]),
                    float(req["destination_y"]),
                )
            except (KeyError, TypeError, ValueError):
                log.warning(
                    "CompassApp returned a result without parseable request O-D; "
                    "skipping this response."
                )
                continue

            od_info = far_rows_by_key.get(od_key)
            if od_info is None:
                log.warning(
                    f"CompassApp returned an unmatched O-D result for key {od_key}; "
                    "skipping this response."
                )
                continue
            processed_keys.add(od_key)

            if "error" in result or result.get("route") is None:
                cp_error = result.get("error", "No route found")
                log.warning(
                    f"CompassApp failed for od_key {od_key}: {cp_error}. "
                    "Creating a straight-line fallback route."
                )
                rows = route_single_trip_fallback(
                    od_info["origin_x"],
                    od_info["origin_y"],
                    od_info["dest_x"],
                    od_info["dest_y"],
                    od_key,
                )
                shape_rows.extend(rows)
                continue

            line = geometry_from_route(result["route"])

            if line.is_empty:
                log.warning(
                    f"CompassApp returned an empty route for od_key {od_key}. "
                    "Creating a straight-line fallback route."
                )
                rows = route_single_trip_fallback(
                    od_info["origin_x"],
                    od_info["origin_y"],
                    od_info["dest_x"],
                    od_info["dest_y"],
                    od_key,
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
                        "shape_id": od_key,  # Use od_key as shape_id
                        "shape_pt_sequence": int(seq),
                        "shape_pt_lon": float(lon),
                        "shape_pt_lat": float(lat),
                        "shape_dist_traveled": float(cum_km),
                    }
                )
                prev_lat, prev_lon = lat, lon

        # Ensure every requested far O-D gets a shape, even if Compass omitted
        # a response or returned an unparsable one.
        missing_keys = set(far_rows_by_key) - processed_keys
        for od_key in missing_keys:
            od_info = far_rows_by_key[od_key]
            log.warning(
                f"CompassApp returned no usable result for od_key {od_key}. "
                "Creating a straight-line fallback route."
            )
            rows = route_single_trip_fallback(
                od_info["origin_x"],
                od_info["origin_y"],
                od_info["dest_x"],
                od_info["dest_y"],
                od_key,
            )
            shape_rows.extend(rows)

    # Step 6: Build output DataFrames
    shapes_df = pd.DataFrame(
        shape_rows,
        columns=[
            "shape_id",
            "shape_pt_sequence",
            "shape_pt_lon",
            "shape_pt_lat",
            "shape_dist_traveled",
        ],
    )

    # Create mapping from block_id to shape_id (via od_key)
    od_mapping_df = od_df[["block_id", "od_key"]].copy()
    od_mapping_df["shape_id"] = od_mapping_df["od_key"]

    return shapes_df, od_mapping_df
