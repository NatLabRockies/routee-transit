import math

# single-global-graph approach; per-row cached graphs removed
import multiprocessing as mp
from typing import Any, Tuple

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union

# Optionally hold a single pre-fetched graph for the whole study area.
GLOBAL_GRAPH = None


def _parallel_map(
    func: Any, iterable: Any, n_processes: int | None, chunksize: int = 8
) -> list[Any]:
    """Simple wrapper to run func over iterable using multiprocessing when requested."""
    if not n_processes or n_processes <= 1:
        return [func(x) for x in iterable]
    results = []
    with mp.Pool(n_processes) as pool:
        for item in pool.imap(func, iterable, chunksize=chunksize):
            results.append(item)
    return results


def _haversine_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> float:
    """Get the haversine distance between two points in kilometers."""
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


def _process_deadhead_trip_row(args: Tuple[Any, ...]) -> list[Any]:
    """Worker that computes shortest-path and returns per-point shape rows.

    args: (idx, start_x, start_y, end_x, end_y, block_id, network_type, road_buffer_m)

    Returns a list of dicts with fields:
      - 'shape_id' (block id)
      - 'shape_pt_sequence' (1-based int)
      - 'shape_pt_lon'
      - 'shape_pt_lat'
      - 'shape_dist_traveled' (cumulative km from route start)
    """
    (
        start_x,
        start_y,
        end_x,
        end_y,
        block_id,
    ) = args

    G = GLOBAL_GRAPH

    src = ox.nearest_nodes(G, start_x, start_y)
    dst = ox.nearest_nodes(G, end_x, end_y)
    route_nodes = nx.shortest_path(G, src, dst, weight="length")

    edge_geoms = []
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        data = G.get_edge_data(u, v)
        if not data:
            raise ValueError(f"Edge {(u, v)} is missing data.")
        key = list(data.keys())[0]
        attr = data[key]
        geom = attr.get("geometry")
        if geom is None:
            ux = G.nodes[u].get("x")
            uy = G.nodes[u].get("y")
            vx = G.nodes[v].get("x")
            vy = G.nodes[v].get("y")
            geom = LineString([(ux, uy), (vx, vy)])
        edge_geoms.append(geom)

    if not edge_geoms:  # Return a link from start to end if no route found
        # return []
        return [
            {
                "shape_id": block_id,
                "shape_pt_sequence": 1,
                "shape_pt_lon": float(start_x),
                "shape_pt_lat": float(start_y),
                "shape_dist_traveled": 0.0,
            },
            {
                "shape_id": block_id,
                "shape_pt_sequence": 2,
                "shape_pt_lon": float(end_x),
                "shape_pt_lat": float(end_y),
                "shape_dist_traveled": _haversine_km(start_y, start_x, end_y, end_x),
            },
        ]

    try:
        merged = linemerge(edge_geoms)
    except Exception:
        merged = unary_union(edge_geoms)
    if getattr(merged, "geom_type", "") == "MultiLineString":
        coords = []
        for part in merged:
            coords.extend(list(part.coords))
        merged = LineString(coords)

    coords = list(merged.coords)  # list of (lon, lat)
    rows = []
    prev_lat = None
    prev_lon = None
    cum_km = 0.0
    for seq, (lon, lat) in enumerate(coords, start=1):
        if prev_lat is not None:
            seg_km = _haversine_km(prev_lat, prev_lon, lat, lon)
            cum_km += seg_km
        else:
            cum_km = 0.0
        rows.append(
            {
                "shape_id": block_id,
                "shape_pt_sequence": int(seq),
                "shape_pt_lon": float(lon),
                "shape_pt_lat": float(lat),
                "shape_dist_traveled": float(cum_km),
            }
        )
        prev_lat, prev_lon = lat, lon

    return rows


def create_deadhead_shapes(
    df: gpd.GeoDataFrame,
    o_col: str = "geometry_origin",
    d_col: str = "geometry_destination",
    n_processes: int | None = None,
) -> pd.DataFrame:
    """Compute deadhead route shapes between origin and destination.

    For each row in `df` this computes a shortest-path on the OSM network between the
    origin (o_col) and destination (d_col) using a single pre-fetched study-area graph
    (fetched from `bbox`). Returns a pandas DataFrame with per-point GTFS-like shape
    rows: ['shape_id', 'shape_pt_sequence', 'shape_pt_lon', 'shape_pt_lat', 'shape_dist_traveled']
    where `shape_dist_traveled` is cumulative distance in kilometers from the route start.
    """
    # Require bbox: this function uses a single study-area graph for all routing.
    global GLOBAL_GRAPH

    # Create bounding box around for osmnx graph based on O/D geometry
    all_points = pd.concat([df["geometry_origin"], df["geometry_destination"]])
    lons = all_points.apply(lambda p: p.x)
    lats = all_points.apply(lambda p: p.y)
    min_lon, max_lon = lons.min(), lons.max()  # Bounding box
    min_lat, max_lat = lats.min(), lats.max()  # Bounding box
    buffer_deg_lat = 0.018  # Roughly 2 km buffer in degrees
    buffer_deg_lon = 0.022  # Roughly 2 km buffer in degrees
    bbox = (
        min_lon - buffer_deg_lon,
        min_lat - buffer_deg_lat,
        max_lon + buffer_deg_lon,
        max_lat + buffer_deg_lat,
    )  # bounding box as (min_x, min_y, max_x, max_y)

    GLOBAL_GRAPH = ox.graph_from_bbox(bbox, network_type="drive")
    # Have OSMNX project to a suitable CRS
    GLOBAL_GRAPH = ox.project_graph(GLOBAL_GRAPH)

    task_args = []
    for _, r in df.iterrows():
        origin = r[o_col]
        destination = r[d_col]
        task_args.append(
            (
                float(origin.x),
                float(origin.y),
                float(destination.x),
                float(destination.y),
                r.get("block_id"),
            )
        )

    results = _parallel_map(_process_deadhead_trip_row, task_args, n_processes)

    # results is a list of lists (per-route rows). Flatten and build DataFrame
    shape_rows = []
    for route_rows in results:
        if not route_rows:
            continue
        # route_rows is a list of dicts from the worker
        shape_rows.extend(route_rows)

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
