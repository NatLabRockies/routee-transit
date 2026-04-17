import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd
import pyproj
import shapely
import shapely.geometry
from nrel.routee.compass import CompassApp
from nrel.routee.compass.map_matching.utils import match_result_to_geopandas

# WGS84 geodesic calculator (module-level singleton for reuse)
_GEOD = pyproj.Geod(ellps="WGS84")

# Dwell detection thresholds
DWELL_DISTANCE_THRESHOLD_M = 10.0  # meters
DWELL_TIME_THRESHOLD_S = 15.0  # seconds


def _parse_maxspeed_mph(val: object) -> float:
    """Parse an OSM maxspeed value to mph. Returns NaN if unparseable."""
    if val is None:
        return float("nan")
    if isinstance(val, list):
        parsed = [_parse_maxspeed_mph(v) for v in val]
        valid = [v for v in parsed if not np.isnan(v)]
        return min(valid) if valid else float("nan")
    if isinstance(val, (int, float)):
        return float(val) if not np.isnan(float(val)) else float("nan")
    s = str(val).strip().lower()
    if s in ("", "none", "signals", "variable", "walk", "national"):
        return float("nan")
    m = re.match(r"(\d+(?:\.\d+)?)\s*(mph|km/h|kph|kmh)?", s)
    if m:
        speed = float(m.group(1))
        unit = (m.group(2) or "mph").lower().replace("/", "")
        if unit in ("kmh", "kph"):
            speed *= 0.621371
        return speed
    return float("nan")


def _parse_highway(val: object) -> str | None:
    """Return the OSM highway tag as a string (normalises list values)."""
    if val is None:
        return None
    if isinstance(val, list):
        return str(val[0]) if val else None
    return str(val)


def _parse_lanes(val: object) -> float:
    """Parse an OSM lanes value to a float. Returns NaN if unparseable."""
    if val is None:
        return float("nan")
    if isinstance(val, list):
        nums = []
        for v in val:
            try:
                nums.append(int(v))
            except (ValueError, TypeError):
                pass
        return float(max(nums)) if nums else float("nan")
    try:
        return float(int(val))
    except (ValueError, TypeError):
        return float("nan")


def geodesic_line_length_m(line_geom: shapely.geometry.LineString) -> float:
    """Compute geodesic length of a LineString in meters using the WGS84 ellipsoid."""
    coords = shapely.get_coordinates(line_geom)
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for i in range(len(coords) - 1):
        _, _, dist = _GEOD.inv(
            coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1]
        )
        total += dist
    return total


def read_realtime_records(path_to_json):
    """Read and flatten GTFS-RT vehicle position records from a JSONL file."""
    records = []
    with open(path_to_json, "r") as f:
        for line in f:
            record = json.loads(line)
            records.append(record)

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

    # Zero lat/lon values sometimes come up — treat as NA
    zero_mask = (df["latitude"] == 0) | (df["longitude"] == 0)
    if zero_mask.any():
        print(f"Replacing {zero_mask.sum()} zero lat/lon values with NA")
    df.loc[zero_mask, ["latitude", "longitude"]] = np.nan

    return df


def clean_trip_df(trip_rt_df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and sort chronologically for a single trip.

    Parameters
    ----------
    trip_rt_df : pd.DataFrame
        DataFrame with realtime observations for a single bus trip.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame sorted by timestamp with duplicates removed.
    """
    trip_rt_df = trip_rt_df.dropna(subset=["timestamp"]).copy()
    trip_rt_df["timestamp"] = pd.to_datetime(
        trip_rt_df["timestamp"].astype(int), unit="s"
    )
    keep_cols = [
        "timestamp",
        "latitude",
        "longitude",
        "current_stop_sequence",
        "stop_id",
    ]
    keep_cols = [c for c in keep_cols if c in trip_rt_df.columns]
    trip_rt_filt = trip_rt_df.drop_duplicates(subset=keep_cols)
    trip_rt_filt = trip_rt_filt.sort_values("timestamp").reset_index(drop=True)
    return trip_rt_filt


def build_compass_app(
    shapes_df: pd.DataFrame,
    buffer_deg: float = 0.05,
) -> tuple[CompassApp, pd.DataFrame]:
    """Build a CompassApp from the bounding box of GTFS shapes.

    Parameters
    ----------
    shapes_df : pd.DataFrame
        GTFS shapes DataFrame with columns 'shape_pt_lat' and 'shape_pt_lon'.
    buffer_deg : float
        Buffer in degrees to add around the bounding box.

    Returns
    -------
    app : CompassApp
        Initialized CompassApp for map matching.
    edge_attr_df : pd.DataFrame
        OSM edge attributes (highway, maxspeed_mph, lanes, grade, grade_abs)
        indexed by Compass edge_id (int). Grade columns are populated by
        Compass, which downloads SRTM elevation data during ``from_graph``.
    """
    min_lat = shapes_df["shape_pt_lat"].min()
    max_lat = shapes_df["shape_pt_lat"].max()
    min_lon = shapes_df["shape_pt_lon"].min()
    max_lon = shapes_df["shape_pt_lon"].max()

    bbox = (
        min_lon - buffer_deg,
        min_lat - buffer_deg,
        max_lon + buffer_deg,
        max_lat + buffer_deg,
    )
    print(f"Building CompassApp from GTFS shapes bounding box: {bbox}")

    graph = ox.graph_from_bbox(bbox=bbox, network_type="drive")

    # Compass downloads SRTM elevation data and calls ox.add_edge_grades
    # internally, mutating the graph in-place before returning the app.
    app = CompassApp.from_graph(graph)

    # Build edge attribute lookup keyed by Compass edge_id.
    # Must happen AFTER from_graph so grade/grade_abs are present on the graph.
    # Compass assigns sequential edge IDs via enumerate(graph.edges()), so we
    # iterate in the same order to get a direct eid → OSM attributes mapping.
    attr_records: dict[int, dict[str, object]] = {}
    for eid, (_u, _v, _key, data) in enumerate(graph.edges(data=True, keys=True)):
        attrs: dict[str, object] = {
            "highway": _parse_highway(data.get("highway")),
            "maxspeed_mph": _parse_maxspeed_mph(data.get("maxspeed")),
            "lanes": _parse_lanes(data.get("lanes")),
        }
        g = data.get("grade")
        attrs["grade"] = float(g) if g is not None and pd.notna(g) else float("nan")
        g = data.get("grade_abs")
        attrs["grade_abs"] = float(g) if g is not None and pd.notna(g) else float("nan")
        attr_records[eid] = attrs

    edge_attr_df = pd.DataFrame.from_dict(attr_records, orient="index")
    edge_attr_df.index.name = "edge_id"

    return app, edge_attr_df


def match_realtime_trip(
    trip_rt_df: pd.DataFrame,
    app: CompassApp,
    edge_attr_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match realtime observations to the OSM network and compute route positions.

    Uses the map matcher's ordered edge sequence as the source of truth for
    observation-to-link assignment.  Each observation is projected onto the matched
    route to obtain a cumulative distance, enabling accurate timestamp interpolation
    at link boundaries.

    Parameters
    ----------
    trip_rt_df : pd.DataFrame
        Cleaned, chronologically sorted GTFS-RT observations for a single trip.
        Must have columns: latitude, longitude, timestamp.
    app : CompassApp
        Initialized CompassApp for map matching.

    Returns
    -------
    obs_df : pd.DataFrame
        Observations augmented with: edge_idx, road_id, geom, link_length_km,
        cumul_dist_m (position along the matched route in meters).
    edges_df : pd.DataFrame
        GeoDataFrame of matched edges in traversal order with geodesic lengths
        and cumulative start distances.
    """
    valid = trip_rt_df.dropna(subset=["latitude", "longitude"]).copy()
    if len(valid) < 2:
        raise ValueError("Fewer than 2 valid coordinates after dropping NaN lat/lon")

    trace = [
        {"x": float(row["longitude"]), "y": float(row["latitude"])}
        for _, row in valid.iterrows()
    ]

    results = app.map_match([{"trace": trace}])
    result = results[0]

    gdf = match_result_to_geopandas(results)
    if gdf.empty:
        raise ValueError("Map matching returned no results")

    # Compute geodesic edge lengths (accurate regardless of latitude)
    gdf["link_length_m"] = gdf.geometry.apply(geodesic_line_length_m)
    gdf["link_length_km"] = gdf["link_length_m"] / 1000.0

    # Build cumulative distance at the start of each edge
    cumul_start = np.zeros(len(gdf))
    for i in range(1, len(gdf)):
        cumul_start[i] = cumul_start[i - 1] + gdf.iloc[i - 1]["link_length_m"]
    gdf["cumul_start_m"] = cumul_start

    # --- Assign each trace point to a matched edge ---
    # Use point_matches (authoritative per-point edge assignment from the map matcher)
    # and resolve to edge_index with greedy forward matching for loop disambiguation.
    point_matches = result.get("point_matches", [])
    edge_ids = gdf["edge_id"].values
    n_edges = len(edge_ids)

    obs_edge_idx = []
    obs_cumul_dist = []
    last_edge_idx = 0

    for i in range(len(valid)):
        row = valid.iloc[i]
        pt = shapely.geometry.Point(row["longitude"], row["latitude"])

        # Get target edge_id from point_matches if available
        target_edge_id = None
        if i < len(point_matches):
            target_edge_id = point_matches[i].get("edge_id")

        # Search forward in the edge sequence for the target edge_id
        matched_idx = None
        if target_edge_id is not None:
            for ei in range(last_edge_idx, n_edges):
                if edge_ids[ei] == target_edge_id:
                    matched_idx = ei
                    break

        # Fallback: nearest edge forward from current position
        if matched_idx is None:
            best_dist = float("inf")
            for ei in range(last_edge_idx, n_edges):
                d = pt.distance(gdf.geometry.iloc[ei])
                if d < best_dist:
                    best_dist = d
                    matched_idx = ei

        if matched_idx is None:
            matched_idx = last_edge_idx

        # Project observation onto matched edge to get fractional position
        edge_geom = gdf.geometry.iloc[matched_idx]
        frac = edge_geom.project(pt, normalized=True)
        frac = max(0.0, min(1.0, frac))

        dist_along_route = (
            cumul_start[matched_idx] + frac * gdf.iloc[matched_idx]["link_length_m"]
        )

        obs_edge_idx.append(matched_idx)
        obs_cumul_dist.append(dist_along_route)
        last_edge_idx = matched_idx

    valid["edge_idx"] = obs_edge_idx
    valid["cumul_dist_m"] = obs_cumul_dist

    # Enforce monotonicity: the bus can only move forward along the matched route
    valid["cumul_dist_m"] = np.maximum.accumulate(valid["cumul_dist_m"].values)

    # Map edge attributes to observations
    valid["road_id"] = [str(edge_ids[ei]) for ei in obs_edge_idx]
    valid["geom"] = [gdf.geometry.iloc[ei] for ei in obs_edge_idx]
    valid["link_length_km"] = [gdf.iloc[ei]["link_length_km"] for ei in obs_edge_idx]

    # Join OSM road attributes (highway type, speed limit, grade, lanes) onto
    # each matched edge using Compass edge_id as the join key.
    if edge_attr_df is not None:
        for col in edge_attr_df.columns:
            gdf[col] = gdf["edge_id"].map(edge_attr_df[col])

    return valid, gdf


def detect_dwell_time(
    obs_df: pd.DataFrame,
    dist_threshold_m: float = DWELL_DISTANCE_THRESHOLD_M,
    time_threshold_s: float = DWELL_TIME_THRESHOLD_S,
) -> pd.Series:
    """Estimate dwell time per edge from consecutive near-stationary observations.

    A dwell is detected when consecutive observations on the same edge are less than
    ``dist_threshold_m`` apart (along the route) but more than ``time_threshold_s``
    apart in time.

    Parameters
    ----------
    obs_df : pd.DataFrame
        Observations with edge_idx, cumul_dist_m, and timestamp columns.
    dist_threshold_m : float
        Maximum displacement (meters) for an interval to count as dwelling.
    time_threshold_s : float
        Minimum elapsed time (seconds) for an interval to count as dwelling.

    Returns
    -------
    pd.Series
        Dwell time in seconds, indexed by edge_idx.
    """
    dwell_by_edge: dict[int, float] = {}
    for edge_idx, group in obs_df.groupby("edge_idx"):
        if len(group) < 2:
            dwell_by_edge[edge_idx] = 0.0
            continue

        sorted_group = group.sort_values("timestamp")
        dists = sorted_group["cumul_dist_m"].values
        times = sorted_group["timestamp"].values

        dwell = 0.0
        for i in range(1, len(sorted_group)):
            d_dist = dists[i] - dists[i - 1]
            d_time = (times[i] - times[i - 1]) / np.timedelta64(1, "s")
            if d_dist < dist_threshold_m and d_time > time_threshold_s:
                dwell += d_time
        dwell_by_edge[edge_idx] = dwell

    return pd.Series(dwell_by_edge, name="dwell_time_sec")


def estimate_link_speeds(obs_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate average speed for each link in the matched route.

    Interpolates timestamps at link boundaries using cumulative distance along the
    matched route (via ``np.interp``), then computes speed = link_length / transit_time.

    Parameters
    ----------
    obs_df : pd.DataFrame
        Observations with cumul_dist_m and timestamp columns (from match_realtime_trip).
    edges_df : pd.DataFrame
        Matched edges with link_length_m and cumul_start_m (from match_realtime_trip).

    Returns
    -------
    pd.DataFrame
        Per-link speed estimates with quality indicators.
    """
    if len(obs_df) < 2:
        return pd.DataFrame()

    # Work in float seconds for interpolation
    t0 = obs_df["timestamp"].iloc[0]
    obs_dists = obs_df["cumul_dist_m"].values
    obs_times_sec = (obs_df["timestamp"] - t0).dt.total_seconds().values

    # Compute link boundary positions (start of each edge + end of last edge)
    edge_starts = edges_df["cumul_start_m"].values
    total_route_m = edge_starts[-1] + edges_df.iloc[-1]["link_length_m"]
    boundaries = np.append(edge_starts, total_route_m)

    # Interpolate timestamps at each link boundary
    boundary_sec = np.interp(boundaries, obs_dists, obs_times_sec)

    n_edges = len(edges_df)
    link_length_m = edges_df["link_length_m"].values
    link_length_mi = link_length_m / 1609.344

    entry_sec = boundary_sec[:n_edges]
    exit_sec = boundary_sec[1:]
    transit_sec = exit_sec - entry_sec

    # Speed in mph (NaN when transit time is zero or negative)
    mph = np.where(transit_sec > 0, link_length_mi / (transit_sec / 3600), np.nan)

    # Observation count per edge
    obs_counts = obs_df.groupby("edge_idx").size()
    n_obs = np.array([obs_counts.get(i, 0) for i in range(n_edges)], dtype=int)

    # Dwell time per edge
    dwell = detect_dwell_time(obs_df)
    dwell_sec = np.array([dwell.get(i, 0.0) for i in range(n_edges)])

    # Speed excluding dwell time
    moving_sec = transit_sec - dwell_sec
    mph_moving = np.where(moving_sec > 0, link_length_mi / (moving_sec / 3600), np.nan)

    # Convert boundary times back to timestamps for the output
    entry_timestamps = t0 + pd.to_timedelta(entry_sec, unit="s")

    link_summary = pd.DataFrame(
        {
            "road_id": edges_df["edge_id"].astype(str).values,
            "edge_idx": np.arange(n_edges),
            "geom": edges_df.geometry.values,
            "link_length_km": link_length_m / 1000.0,
            "transit_time_sec": transit_sec,
            "mph": mph,
            "dwell_time_sec": dwell_sec,
            "mph_moving": mph_moving,
            "first_timestamp": entry_timestamps,
            "n_observations": n_obs,
            "speed_source": np.where(n_obs >= 2, "observed", "interpolated"),
        }
    )

    # Propagate OSM road attributes from the matched edges GeoDataFrame.
    for col in ("highway", "maxspeed_mph", "lanes", "grade", "grade_abs"):
        if col in edges_df.columns:
            link_summary[col] = edges_df[col].values

    return link_summary


# ---------------------------------------------------------------------------
# GTFS scheduled-speed and stop-count features
# ---------------------------------------------------------------------------


def parse_gtfs_time_to_seconds(time_val: object) -> float:
    """Parse a GTFS time value to total seconds since midnight.

    Handles strings like ``"08:30:00"`` or ``"25:30:00"`` (hours > 24 are
    valid in GTFS for service past midnight), pandas Timedelta objects (as
    returned by gtfsblocks Feed), and plain numeric values already in seconds.

    Returns ``float("nan")`` for missing or unparseable values.
    """
    if time_val is None:
        return float("nan")
    try:
        if pd.isna(time_val):  # type: ignore[arg-type]
            return float("nan")
    except (TypeError, ValueError):
        pass
    if isinstance(time_val, pd.Timedelta):
        return time_val.total_seconds()
    if isinstance(time_val, (int, float)):
        return float(time_val)
    parts = str(time_val).strip().split(":")
    if len(parts) == 3:
        try:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except ValueError:
            pass
    return float("nan")


def project_stops_to_route(
    stop_times_trip: pd.DataFrame,
    stops_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    max_snap_dist_deg: float = 0.05,
) -> pd.DataFrame:
    """Project GTFS stops onto the matched OSM edge sequence for a single trip.

    Each stop is assigned to the nearest forward edge in the matched route
    using the same greedy-forward search as :func:`match_realtime_trip`.  The
    stop's position within that edge is obtained by projecting its WGS84
    coordinates onto the edge geometry, yielding a cumulative distance along
    the matched route.

    ``shape_dist_traveled`` is intentionally ignored — that field has
    non-standard units and is unreliable across GTFS feeds.  All distances are
    derived purely from coordinates.

    Parameters
    ----------
    stop_times_trip : pd.DataFrame
        stop_times.txt rows for a **single trip**, sorted or unsorted.  Must
        have columns ``stop_id``, ``stop_sequence``, and ``departure_time``.
        The column ``arrival_time`` is used when present.
    stops_df : pd.DataFrame
        stops.txt **indexed by stop_id** with columns ``stop_lat`` and
        ``stop_lon`` (WGS84 degrees).
    edges_df : pd.DataFrame
        Matched edges GeoDataFrame from :func:`match_realtime_trip`, with
        columns ``edge_id``, ``geometry`` (LineString, WGS84),
        ``link_length_m``, and ``cumul_start_m``.
    max_snap_dist_deg : float, default 0.05
        Maximum WGS84 Euclidean distance (degrees) between a stop coordinate
        and the nearest matched edge for the stop to be included.  Stops
        farther than this threshold are silently dropped.  At mid-latitudes
        0.05° ≈ 4-5 km, which accepts any plausible GTFS stop while
        rejecting clearly erroneous coordinates.

    Returns
    -------
    pd.DataFrame
        One row per successfully matched stop, columns: ``stop_id``,
        ``stop_sequence``, ``edge_idx``, ``road_id``, ``cumul_dist_m``,
        ``departure_sec``, ``arrival_sec``.  Stops whose ``stop_id`` is absent
        from *stops_df*, or whose nearest edge exceeds *max_snap_dist_deg*,
        are silently dropped.
    """
    _EMPTY = pd.DataFrame(
        columns=[
            "stop_id",
            "stop_sequence",
            "edge_idx",
            "road_id",
            "cumul_dist_m",
            "departure_sec",
            "arrival_sec",
        ]
    )
    if edges_df.empty or stop_times_trip.empty:
        return _EMPTY

    stops_ordered = stop_times_trip.sort_values("stop_sequence")
    edge_ids = edges_df["edge_id"].values
    n_edges = len(edge_ids)
    cumul_start = edges_df["cumul_start_m"].values
    link_lengths = edges_df["link_length_m"].values

    results: list[dict] = []
    last_edge_idx = 0

    for _, row in stops_ordered.iterrows():
        stop_id = row["stop_id"]
        if stop_id not in stops_df.index:
            continue

        stop_row = stops_df.loc[stop_id]
        # stops_df.loc may return a DataFrame when stop_id is duplicated; take first.
        if isinstance(stop_row, pd.DataFrame):
            stop_row = stop_row.iloc[0]

        pt = shapely.geometry.Point(
            float(stop_row["stop_lon"]), float(stop_row["stop_lat"])
        )

        # Greedy-forward search: scan edges from last matched position onward,
        # choose the one whose geometry is nearest to the stop coordinate.
        best_dist = float("inf")
        best_idx = last_edge_idx
        for ei in range(last_edge_idx, n_edges):
            d = pt.distance(edges_df.geometry.iloc[ei])
            if d < best_dist:
                best_dist = d
                best_idx = ei

        # Drop stops that are implausibly far from the matched route.
        if best_dist > max_snap_dist_deg:
            continue

        # Project stop onto the matched edge to get fractional position.
        edge_geom = edges_df.geometry.iloc[best_idx]
        frac = max(0.0, min(1.0, edge_geom.project(pt, normalized=True)))
        cumul_dist_m = cumul_start[best_idx] + frac * link_lengths[best_idx]

        results.append(
            {
                "stop_id": stop_id,
                "stop_sequence": int(row["stop_sequence"]),
                "edge_idx": best_idx,
                "road_id": str(edge_ids[best_idx]),
                "cumul_dist_m": cumul_dist_m,
                "departure_sec": parse_gtfs_time_to_seconds(row.get("departure_time")),
                "arrival_sec": parse_gtfs_time_to_seconds(
                    row.get("arrival_time", float("nan"))
                ),
            }
        )
        last_edge_idx = best_idx

    return pd.DataFrame(results) if results else _EMPTY


def compute_scheduled_speeds_between_stops(
    stops_on_route: pd.DataFrame,
) -> pd.DataFrame:
    """Compute GTFS-scheduled moving speed for each consecutive stop pair.

    Uses arrival-at-B minus departure-from-A timing, which excludes dwell
    time at both endpoints.  This matches the semantics of ``mph_moving`` in
    :func:`estimate_link_speeds`.

    Parameters
    ----------
    stops_on_route : pd.DataFrame
        Output of :func:`project_stops_to_route` for a single trip.

    Returns
    -------
    pd.DataFrame
        One row per consecutive stop pair, columns: ``stop_seq_from``,
        ``stop_seq_to``, ``edge_idx_from``, ``edge_idx_to``,
        ``cumul_dist_from``, ``cumul_dist_to``, ``dist_m``, ``time_sec``,
        ``scheduled_speed_mph``.  ``scheduled_speed_mph`` is NaN when
        ``time_sec`` ≤ 0 or ``dist_m`` ≤ 0.
    """
    _EMPTY_COLS = [
        "stop_seq_from",
        "stop_seq_to",
        "edge_idx_from",
        "edge_idx_to",
        "cumul_dist_from",
        "cumul_dist_to",
        "dist_m",
        "time_sec",
        "scheduled_speed_mph",
    ]
    if len(stops_on_route) < 2:
        return pd.DataFrame(columns=_EMPTY_COLS)

    rows = stops_on_route.sort_values("stop_sequence").reset_index(drop=True)
    results: list[dict] = []

    for i in range(len(rows) - 1):
        a = rows.iloc[i]
        b = rows.iloc[i + 1]

        dist_m = b["cumul_dist_m"] - a["cumul_dist_m"]
        # Arrival-at-B minus departure-from-A: pure travel time excluding
        # dwell at both endpoints.
        time_sec = b["arrival_sec"] - a["departure_sec"]

        if time_sec > 0 and dist_m > 0:
            speed_mph = (dist_m / 1609.344) / (time_sec / 3600.0)
        else:
            speed_mph = float("nan")

        results.append(
            {
                "stop_seq_from": int(a["stop_sequence"]),
                "stop_seq_to": int(b["stop_sequence"]),
                "edge_idx_from": int(a["edge_idx"]),
                "edge_idx_to": int(b["edge_idx"]),
                "cumul_dist_from": float(a["cumul_dist_m"]),
                "cumul_dist_to": float(b["cumul_dist_m"]),
                "dist_m": dist_m,
                "time_sec": time_sec,
                "scheduled_speed_mph": speed_mph,
            }
        )

    return pd.DataFrame(results)


def aggregate_gtfs_features_by_edge(
    stops_on_route: pd.DataFrame,
    sched_speeds: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate GTFS stop and scheduled-speed features to the OSM edge level.

    Parameters
    ----------
    stops_on_route : pd.DataFrame
        Output of :func:`project_stops_to_route`.
    sched_speeds : pd.DataFrame
        Output of :func:`compute_scheduled_speeds_between_stops`.
    edges_df : pd.DataFrame
        Matched edges GeoDataFrame from :func:`match_realtime_trip`, with
        columns ``cumul_start_m`` and ``link_length_m``.

    Returns
    -------
    pd.DataFrame
        Indexed by ``edge_idx`` (0 to *n_edges* − 1), two columns:

        ``n_stops``
            Integer count of scheduled stops projected onto this edge.
        ``scheduled_speed_mph``
            Distance-weighted mean GTFS-scheduled speed (mph) across all
            stop-to-stop segments whose projected range overlaps this edge.
            NaN when no segment covers the edge.
    """
    n_edges = len(edges_df)
    n_stops_arr = np.zeros(n_edges, dtype=int)
    sched_speed_arr = np.full(n_edges, float("nan"))

    if not stops_on_route.empty:
        for ei, count in stops_on_route.groupby("edge_idx").size().items():
            if 0 <= ei < n_edges:
                n_stops_arr[int(ei)] = int(count)

    if not sched_speeds.empty:
        cumul_start = edges_df["cumul_start_m"].values
        link_lengths = edges_df["link_length_m"].values
        weighted_speed_sum = np.zeros(n_edges)
        weight_sum = np.zeros(n_edges)

        for _, seg in sched_speeds.iterrows():
            if pd.isna(seg["scheduled_speed_mph"]):
                continue
            ei_from = int(seg["edge_idx_from"])
            ei_to = int(seg["edge_idx_to"])
            seg_start = float(seg["cumul_dist_from"])
            seg_end = float(seg["cumul_dist_to"])
            spd = float(seg["scheduled_speed_mph"])
            for ei in range(max(0, ei_from), min(n_edges, ei_to + 1)):
                edge_start = cumul_start[ei]
                edge_end = edge_start + link_lengths[ei]
                overlap = min(seg_end, edge_end) - max(seg_start, edge_start)
                if overlap > 0:
                    weighted_speed_sum[ei] += overlap * spd
                    weight_sum[ei] += overlap

        nonzero = weight_sum > 0
        sched_speed_arr[nonzero] = weighted_speed_sum[nonzero] / weight_sum[nonzero]

    return pd.DataFrame(
        {"n_stops": n_stops_arr, "scheduled_speed_mph": sched_speed_arr},
        index=pd.RangeIndex(n_edges, name="edge_idx"),
    )


def get_link_speeds_for_trip(
    trip_id: str,
    rt_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    app: CompassApp,
    edge_attr_df: pd.DataFrame | None = None,
    stop_times_df: pd.DataFrame | None = None,
    stops_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute per-link speed estimates for a single trip.

    Parameters
    ----------
    trip_id : str
        GTFS trip ID.
    rt_df : pd.DataFrame
        All realtime records (will be filtered to this trip).
    trips_df : pd.DataFrame
        GTFS trips.txt indexed by trip_id.
    app : CompassApp
        Initialized CompassApp for map matching.
    edge_attr_df : pd.DataFrame | None, optional
        OSM edge attributes indexed by Compass edge_id (from
        :func:`build_compass_app`).
    stop_times_df : pd.DataFrame | None, optional
        stop_times.txt with a ``trip_id`` column (not set as index).  When
        provided alongside *stops_df*, GTFS features ``n_stops`` and
        ``scheduled_speed_mph`` are computed and added to each link row.
    stops_df : pd.DataFrame | None, optional
        stops.txt **indexed by stop_id** with columns ``stop_lat`` /
        ``stop_lon`` (WGS84).

    Returns
    -------
    pd.DataFrame
        Per-link speed estimates, or empty DataFrame if the trip can't be processed.
    """
    trip_df = rt_df[rt_df["trip_id"] == trip_id].copy()
    trip_df_filt = clean_trip_df(trip_df)

    if len(trip_df_filt) < 10:
        logging.debug(
            "Trip %s skipped: only %d observations after cleaning",
            trip_id,
            len(trip_df_filt),
        )
        return pd.DataFrame(columns=["_skip_reason"]).assign(
            _skip_reason=f"too_few_obs:{len(trip_df_filt)}"
        )[:0]

    has_rt_speed = "speed" in trip_df_filt.columns

    try:
        obs_df, edges_df = match_realtime_trip(
            trip_df_filt, app, edge_attr_df=edge_attr_df
        )
    except (ValueError, shapely.errors.GEOSException) as exc:
        logging.debug("Trip %s skipped: map match error: %s", trip_id, exc)
        return pd.DataFrame()

    link_summary = estimate_link_speeds(obs_df, edges_df)
    if link_summary.empty:
        return pd.DataFrame()

    # Attach trip metadata
    link_summary["trip_id"] = trip_id
    link_summary["route_id"] = trip_df_filt["route_id"].iloc[0]
    link_summary["vehicle_id"] = trip_df_filt["vehicle_id"].iloc[0]

    # Attach median GTFS-RT reported speed per link for verification
    if has_rt_speed:
        rt_speed_by_edge = (
            obs_df.dropna(subset=["speed"]).groupby("edge_idx")["speed"].median()
        )
        link_summary["rt_speed_mps"] = link_summary["edge_idx"].map(rt_speed_by_edge)

    # Attach GTFS stop features: n_stops and scheduled speed per edge.
    if stop_times_df is not None and stops_df is not None:
        trip_stop_times = stop_times_df[stop_times_df["trip_id"] == trip_id]
        if not trip_stop_times.empty and not edges_df.empty:
            try:
                stops_on_route = project_stops_to_route(
                    trip_stop_times, stops_df, edges_df
                )
                sched_speeds = compute_scheduled_speeds_between_stops(stops_on_route)
                gtfs_feats = aggregate_gtfs_features_by_edge(
                    stops_on_route, sched_speeds, edges_df
                )
                link_summary["n_stops"] = gtfs_feats["n_stops"].values
                link_summary["scheduled_speed_mph"] = gtfs_feats[
                    "scheduled_speed_mph"
                ].values
            except Exception as exc:
                logging.debug(
                    "Trip %s: GTFS feature computation failed: %s", trip_id, exc
                )

    return link_summary


def aggregate_speeds_across_trips(all_trip_speeds: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-trip link speeds into a single estimate per road link.

    Uses observation-count-weighted average across trips for each unique road segment.

    Parameters
    ----------
    all_trip_speeds : pd.DataFrame
        Concatenated per-trip link speed estimates.

    Returns
    -------
    pd.DataFrame
        One row per unique road_id with weighted-average speed.
    """
    valid = all_trip_speeds.dropna(subset=["mph"]).copy()
    valid = valid[np.isfinite(valid["mph"])]

    if valid.empty:
        return pd.DataFrame()

    def weighted_mean(group: pd.DataFrame) -> pd.Series:
        weights = group["n_observations"].values.astype(float)
        total_weight = weights.sum()
        if total_weight == 0:
            weights = np.ones(len(group))
            total_weight = float(len(group))

        wmean_mph = np.average(group["mph"].values, weights=weights)

        wmean_moving = np.nan
        moving_valid = group["mph_moving"].dropna()
        if len(moving_valid) > 0:
            w_moving = weights[group["mph_moving"].notna()]
            if w_moving.sum() == 0:
                w_moving = np.ones(len(moving_valid))
            wmean_moving = np.average(moving_valid.values, weights=w_moving)

        return pd.Series(
            {
                "mph_mean": wmean_mph,
                "mph_moving_mean": wmean_moving,
                "geom": group["geom"].iloc[0],
                "link_length_km": group["link_length_km"].iloc[0],
                "n_trips": len(group),
                "total_observations": int(total_weight),
            }
        )

    aggregated = (
        valid.groupby("road_id")
        .apply(weighted_mean, include_groups=False)
        .reset_index()
    )

    # Road properties are constant per road_id — attach from first occurrence.
    road_prop_cols = [
        c
        for c in valid.columns
        if c in {"highway", "maxspeed_mph", "lanes", "grade", "grade_abs"}
    ]
    if road_prop_cols:
        road_props = valid.groupby("road_id")[road_prop_cols].first().reset_index()
        aggregated = aggregated.merge(road_props, on="road_id", how="left")

    return aggregated


def get_speeds_for_one_day(
    path_to_json: Path | os.PathLike,
):
    if not isinstance(path_to_json, Path):
        path_to_json = Path(path_to_json)

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

    # Load stop data for GTFS features (optional — graceful fallback when absent).
    stop_times_df: pd.DataFrame | None = None
    stops_df: pd.DataFrame | None = None
    try:
        stop_times_df = pd.read_csv(
            gtfs_root / "static/stop_times.txt",
            dtype={"trip_id": str, "stop_id": str},
        )
        stops_df = pd.read_csv(
            gtfs_root / "static/stops.txt",
            dtype={"stop_id": str},
        ).set_index("stop_id")
        print(
            f"Loaded stop data: {len(stop_times_df):,} stop_times rows, "
            f"{len(stops_df):,} stops"
        )
    except FileNotFoundError:
        print("stop_times.txt or stops.txt not found — GTFS stop features will be NaN")

    # Read in realtime data
    rt_df = read_realtime_records(path_to_json)
    print(f"{rt_df.trip_id.nunique()} trips on this day")

    # If route IDs aren't included in realtime data, merge them in from trips.txt.
    if "route_id" not in rt_df.columns:
        rt_df = rt_df.merge(trips_df[["route_id"]], left_on="trip_id", right_index=True)

    # Build the CompassApp once using the GTFS shapes bounding box (clean, authoritative)
    app, edge_attr_df = build_compass_app(shapes_df)

    first_trip_start = time.time()
    all_results = []

    for ix, (route_id, route_rt) in enumerate(rt_df.groupby("route_id")):
        trip_ids = list(route_rt["trip_id"].unique())
        print(f"Analyzing {len(trip_ids)} trips on route {route_id}")
        route_start = time.time()
        raw_results = [
            get_link_speeds_for_trip(
                tid,
                rt_df=rt_df,
                trips_df=trips_df,
                app=app,
                edge_attr_df=edge_attr_df,
                stop_times_df=stop_times_df,
                stops_df=stops_df,
            )
            for tid in trip_ids
        ]
        n_skipped = sum(
            1 for r in raw_results if r.empty or ("_skip_reason" in r.columns)
        )
        results = [
            r for r in raw_results if not r.empty and "_skip_reason" not in r.columns
        ]
        if n_skipped:
            # Sample a few raw observations to help diagnose sparse/missing data
            sample_tid = trip_ids[0]
            sample_raw = rt_df[rt_df["trip_id"] == sample_tid]
            sample_clean = clean_trip_df(sample_raw.copy())
            print(
                f"  ⚠ {n_skipped}/{len(trip_ids)} trips skipped on route {route_id}. "
                f"Sample trip '{sample_tid}': "
                f"{len(sample_raw)} raw rows → {len(sample_clean)} after cleaning"
                + (
                    f" (need ≥10; lat/lon null: {sample_raw[['latitude', 'longitude']].isna().any(axis=1).sum()})"
                    if not sample_raw.empty
                    else ""
                )
            )
        if results:
            route_df = pd.concat(results, ignore_index=True)
            route_df.to_csv(gtfs_root / f"realtime_speeds_{route_id}.csv")
            all_results.append(route_df)
        print(
            f"Analyzing {route_rt.trip_id.nunique()} trips on route {route_id} "
            f"took {time.time() - route_start:.2f} s"
        )
        print(f"Finished analyzing {ix + 1} of {rt_df.route_id.nunique()} routes")

    # Concatenate all the files previously written for each route. Save them as a new
    # file and delete the smaller files.
    file_date = str(path_to_json).split(".")[0].split("_")[-1]
    all_csvs = list(gtfs_root.glob("realtime_speeds_*.csv"))
    if all_csvs:
        combined_df = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
        all_speeds_file = gtfs_root / f"realtime_link_speeds_{file_date}.csv"
        combined_df.to_csv(all_speeds_file, index=False)
        for f in all_csvs:
            os.remove(f)
        print(f"Combined all route CSVs into {all_speeds_file} and deleted originals.")

    # Aggregate speeds across trips (weighted average per road segment)
    if all_results:
        all_trips_df = pd.concat(all_results, ignore_index=True)
        aggregated = aggregate_speeds_across_trips(all_trips_df)
        if not aggregated.empty:
            agg_file = gtfs_root / f"realtime_link_speeds_aggregated_{file_date}.csv"
            aggregated.to_csv(agg_file, index=False)
            print(f"Aggregated link speeds saved to {agg_file}")

    print(
        f"Analyzing all {rt_df.trip_id.nunique()} trips "
        f"took {time.time() - first_trip_start:.2f} s"
    )


if __name__ == "__main__":
    # json_path = "reports/realtime/greater_portland_me/gtfs_realtime_records_20251023.jsonl"
    json_path = "reports/realtime/kingcounty/gtfs_realtime_records_20251029.jsonl"
    get_speeds_for_one_day(json_path)
