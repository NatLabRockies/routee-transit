import logging

import geopandas as gpd
import pandas as pd

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
    app : CompassApp or None
        The CompassApp instance, loaded lazily on first use
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
        self.app = None

    def _route_single_trip_fallback(
        self,
        start_lon: float,
        start_lat: float,
        end_lon: float,
        end_lat: float,
        block_id: str,
    ) -> list[dict]:
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

    def _ensure_app_loaded(self) -> None:
        """Lazy-load the CompassApp if not already loaded."""
        if self.app is None:
            import osmnx as ox

            graph = ox.graph_from_bbox(self.bbox, network_type=self.network_type)
            self.app = CompassApp.from_graph(graph)

    def create_deadhead_shapes(
        self,
        df: gpd.GeoDataFrame,
        o_col: str = "geometry_origin",
        d_col: str = "geometry_destination",
        n_processes: int | None = None,
    ) -> pd.DataFrame:
        """
        Compute deadhead route shapes between origin and destination.

        For each row in `df`, this computes a shortest-path using CompassApp.
        Returns a pandas DataFrame with per-point GTFS-like shape rows.

        Parameters
        ----------
        df : gpd.GeoDataFrame
            DataFrame with origin and destination geometry columns
        o_col : str, optional
            Column name for origin geometries (default: "geometry_origin")
        d_col : str, optional
            Column name for destination geometries (default: "geometry_destination")
        n_processes : int or None, optional
            Ignored. CompassApp handles parallelization internally.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ['shape_id', 'shape_pt_sequence', 'shape_pt_lon',
            'shape_pt_lat', 'shape_dist_traveled'] where `shape_dist_traveled` is
            cumulative distance in kilometers from the route start.
        """
        self._ensure_app_loaded()
        assert self.app is not None

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
        results = self.app.run(queries)
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
                rows = self._route_single_trip_fallback(
                    origin.x, origin.y, destination.x, destination.y, block_id
                )
                shape_rows.extend(rows)
                continue

            line = geometry_from_route(result["route"])
            coords = list(line.coords)
            prev_lat, prev_lon = None, None
            cum_km = 0.0
            for seq, (lon, lat) in enumerate(coords, start=1):
                if prev_lat is not None:
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
