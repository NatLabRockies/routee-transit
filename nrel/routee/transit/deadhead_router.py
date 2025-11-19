import math
import multiprocessing as mp
from functools import partial
from typing import Any

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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


class NetworkRouter:
    """Manages OSM network routing for deadhead trip trace generation.

    This class encapsulates the OSM network graph and provides methods for
    computing shortest-path routes between origin-destination pairs.

    Attributes
    ----------
    bbox : tuple[float, float, float, float]
        Bounding box as (min_lon, min_lat, max_lon, max_lat)
    network_type : str
        OSMnx network type (e.g., "drive", "walk", "bike")
    graph : nx.MultiDiGraph or None
        The OSM network graph, loaded lazily on first use
    """

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        network_type: str = "drive",
    ):
        """
        Initialize router with a bounding box.

        Parameters
        ----------
        bbox : tuple[float, float, float, float]
            Bounding box as (min_lon, min_lat, max_lon, max_lat)
        network_type : str, optional
            OSMnx network type (default: "drive")
        """
        self.bbox = bbox
        self.network_type = network_type
        self.graph = None

    def _route_single_trip(
        self,
        start_lon: float,
        start_lat: float,
        end_lon: float,
        end_lat: float,
        block_id: str,
    ) -> list[dict[str, Any]]:
        """
        Compute shortest-path route for a single origin-destination pair.

        This function finds the shortest path on the OSM network graph between
        the given origin and destination coordinates, then returns a list of
        shape points suitable for GTFS shapes.txt format.

        Parameters
        ----------
        start_lon : float
            Origin longitude
        start_lat : float
            Origin latitude
        end_lon : float
            Destination longitude
        end_lat : float
            Destination latitude
        block_id : str
            Block identifier used as shape_id

        Returns
        -------
        list[dict[str, Any]]
            List of dicts with fields:
            - 'shape_id' (str): Block identifier
            - 'shape_pt_sequence' (int): 1-based sequence number
            - 'shape_pt_lon' (float): Longitude
            - 'shape_pt_lat' (float): Latitude
            - 'shape_dist_traveled' (float): Cumulative distance in km from route start
        """
        assert self.graph is not None
        graph = self.graph
        src = ox.nearest_nodes(graph, start_lon, start_lat)
        dst = ox.nearest_nodes(graph, end_lon, end_lat)
        route_nodes = nx.shortest_path(graph, src, dst, weight="length")

        edge_geoms = []
        for u, v in zip(route_nodes[:-1], route_nodes[1:]):
            data = graph.get_edge_data(u, v)
            if not data:
                raise ValueError(f"Edge {(u, v)} is missing data.")
            key = list(data.keys())[0]
            attr = data[key]
            geom = attr.get("geometry")
            if geom is None:
                ux = graph.nodes[u].get("x")
                uy = graph.nodes[u].get("y")
                vx = graph.nodes[v].get("x")
                vy = graph.nodes[v].get("y")
                geom = LineString([(ux, uy), (vx, vy)])
            edge_geoms.append(geom)

        if not edge_geoms:  # Return a link from start to end if no route found
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
        prev_lon = 0.0
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

    @classmethod
    def from_geometries(
        cls,
        geometries: pd.Series,
        buffer_deg_lat: float = 0.018,
        buffer_deg_lon: float = 0.022,
        network_type: str = "drive",
    ) -> "NetworkRouter":
        """
        Create router from a collection of point geometries.

        Automatically computes bounding box with buffer around the geometries.

        Parameters
        ----------
        geometries : pd.Series
            Series of shapely Point geometries
        buffer_deg_lat : float, optional
            Latitude buffer in degrees (default: 0.018, roughly 2 km)
        buffer_deg_lon : float, optional
            Longitude buffer in degrees (default: 0.022, roughly 2 km)
        network_type : str, optional
            OSMnx network type (default: "drive")

        Returns
        -------
        NetworkRouter
            New router instance with computed bounding box
        """
        lons = geometries.apply(lambda p: p.x)
        lats = geometries.apply(lambda p: p.y)
        min_lon, max_lon = lons.min(), lons.max()
        min_lat, max_lat = lats.min(), lats.max()

        bbox = (
            min_lon - buffer_deg_lon,
            min_lat - buffer_deg_lat,
            max_lon + buffer_deg_lon,
            max_lat + buffer_deg_lat,
        )

        return cls(bbox, network_type)

    def _ensure_graph_loaded(self) -> None:
        """Lazy-load the OSM graph if not already loaded."""
        if self.graph is None:
            self.graph = ox.graph_from_bbox(self.bbox, network_type=self.network_type)
            self.graph = ox.project_graph(self.graph)

    def create_deadhead_shapes(
        self,
        df: gpd.GeoDataFrame,
        o_col: str = "geometry_origin",
        d_col: str = "geometry_destination",
        n_processes: int | None = None,
    ) -> pd.DataFrame:
        """
        Compute deadhead route shapes between origin and destination.

        For each row in `df`, this computes a shortest-path on the OSM network
        between the origin (o_col) and destination (d_col). Returns a pandas
        DataFrame with per-point GTFS-like shape rows.

        Parameters
        ----------
        df : gpd.GeoDataFrame
            DataFrame with origin and destination geometry columns
        o_col : str, optional
            Column name for origin geometries (default: "geometry_origin")
        d_col : str, optional
            Column name for destination geometries (default: "geometry_destination")
        n_processes : int or None, optional
            Number of processes for parallel routing. If None or 1, runs serially.
            Serial processing is recommended for small datasets.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ['shape_id', 'shape_pt_sequence', 'shape_pt_lon',
            'shape_pt_lat', 'shape_dist_traveled'] where `shape_dist_traveled` is
            cumulative distance in kilometers from the route start.
        """
        self._ensure_graph_loaded()

        # Prepare task arguments - each is a tuple of (start_lon, start_lat, end_lon, end_lat, block_id)
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

        # Create a partial function with graph pre-bound
        route_func = partial(self._route_single_trip)

        # Process routes (serial or parallel)
        if not n_processes or n_processes <= 1:
            # Serial processing
            results = [route_func(*args) for args in task_args]
        else:
            # Parallel processing using multiprocessing.Pool with starmap
            with mp.Pool(n_processes) as pool:
                results = pool.starmap(route_func, task_args, chunksize=8)

        # Flatten results and build DataFrame
        shape_rows = []
        for route_rows in results:
            if route_rows:
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
