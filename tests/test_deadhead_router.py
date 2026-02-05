import unittest
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from nrel.routee.transit.deadhead_router import NetworkRouter


class TestNetworkRouter(unittest.TestCase):
    def setUp(self):
        self.bbox = (-105.0, 39.0, -104.9, 39.1)
        self.router = NetworkRouter(self.bbox)

    def test_init(self):
        self.assertEqual(self.router.bbox, self.bbox)
        self.assertEqual(self.router.network_type, "drive")
        self.assertIsNone(self.router.graph)

    def test_from_geometries(self):
        geoms = pd.Series([Point(-105.0, 39.0), Point(-104.8, 39.2)])
        router = NetworkRouter.from_geometries(
            geoms, buffer_deg_lat=0.1, buffer_deg_lon=0.1
        )

        # Expected bbox: (-105.1, 38.9, -104.7, 39.3)
        expected_bbox = (-105.1, 38.9, -104.7, 39.3)
        for act, exp in zip(router.bbox, expected_bbox):
            self.assertAlmostEqual(act, exp)

    @patch("nrel.routee.transit.deadhead_router.ox")
    def test_ensure_graph_loaded(self, mock_ox):
        mock_graph = MagicMock()
        mock_ox.graph_from_bbox.return_value = mock_graph
        mock_ox.project_graph.return_value = mock_graph

        self.router._ensure_graph_loaded()

        mock_ox.graph_from_bbox.assert_called_once_with(self.bbox, network_type="drive")
        mock_ox.project_graph.assert_called_once_with(mock_graph)
        self.assertEqual(self.router.graph, mock_graph)

    @patch("nrel.routee.transit.deadhead_router.ox")
    @patch("nrel.routee.transit.deadhead_router.nx")
    def test_create_deadhead_shapes(self, mock_nx, mock_ox):
        # Setup mock graph
        mock_graph = MagicMock()
        mock_graph.nodes = {1: {"x": -105.0, "y": 39.0}, 2: {"x": -104.9, "y": 39.1}}
        # Edge data
        mock_graph.get_edge_data.return_value = {0: {"length": 1000}}  # No geometry

        self.router.graph = mock_graph
        mock_ox.nearest_nodes.side_effect = [1, 2]  # start node, end node
        mock_nx.shortest_path.return_value = [1, 2]

        df = gpd.GeoDataFrame(
            {
                "block_id": ["B1"],
                "geometry_origin": [Point(-105.0, 39.0)],
                "geometry_destination": [Point(-104.9, 39.1)],
            }
        )

        out_df = self.router.create_deadhead_shapes(df)

        self.assertIsInstance(out_df, pd.DataFrame)
        self.assertTrue(len(out_df) >= 2)
        self.assertEqual(out_df.iloc[0]["shape_id"], "B1")
        self.assertEqual(out_df.iloc[0]["shape_pt_sequence"], 1)

    def test_haversine_km(self):
        from nrel.routee.transit.deadhead_router import _haversine_km

        # Distance between (0,0) and (0,1) lat is approx 111.19 km
        dist = _haversine_km(0, 0, 1, 0)
        self.assertAlmostEqual(dist, 111.19, places=1)

    @patch("nrel.routee.transit.deadhead_router.ox")
    @patch("nrel.routee.transit.deadhead_router.nx")
    def test_route_single_trip_no_route_fallback(self, mock_nx, mock_ox):
        # Test fallback when no edges are found
        mock_graph = MagicMock()
        self.router.graph = mock_graph
        mock_ox.nearest_nodes.side_effect = [1, 2]
        mock_nx.shortest_path.return_value = [1]  # Path of length 1 means no edges

        rows = self.router._route_single_trip(-105.0, 39.0, -104.9, 39.1, "B1")

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["shape_pt_lon"], -105.0)
        self.assertEqual(rows[-1]["shape_pt_lon"], -104.9)
        self.assertGreater(rows[-1]["shape_dist_traveled"], 0)

    def test_haversine_km_edge_cases(self):
        from nrel.routee.transit.deadhead_router import _haversine_km

        # Same point should return 0
        dist = _haversine_km(39.0, -105.0, 39.0, -105.0)
        self.assertAlmostEqual(dist, 0.0, places=5)

        # Equator to equator (1 degree lon) is approx 111.19 km
        dist = _haversine_km(0, 0, 0, 1)
        self.assertAlmostEqual(dist, 111.19, places=1)

    @patch("nrel.routee.transit.deadhead_router.ox")
    @patch("nrel.routee.transit.deadhead_router.nx")
    def test_route_single_trip_with_geometry(self, mock_nx, mock_ox):
        from shapely.geometry import LineString

        # Test with edge that has geometry
        mock_graph = MagicMock()
        geom = LineString([(-105.0, 39.0), (-104.95, 39.05), (-104.9, 39.1)])
        mock_graph.get_edge_data.return_value = {0: {"length": 1000, "geometry": geom}}
        mock_graph.nodes = {1: {"x": -105.0, "y": 39.0}, 2: {"x": -104.9, "y": 39.1}}

        self.router.graph = mock_graph
        mock_ox.nearest_nodes.side_effect = [1, 2]
        mock_nx.shortest_path.return_value = [1, 2]

        rows = self.router._route_single_trip(-105.0, 39.0, -104.9, 39.1, "B1")

        self.assertEqual(len(rows), 3)  # Should have 3 points from the LineString
        self.assertEqual(rows[0]["shape_id"], "B1")
        self.assertEqual(rows[0]["shape_pt_sequence"], 1)
        self.assertEqual(rows[0]["shape_dist_traveled"], 0.0)
        # Last point should have accumulated distance
        self.assertGreater(rows[-1]["shape_dist_traveled"], 0)

    @patch("nrel.routee.transit.deadhead_router.ox")
    @patch("nrel.routee.transit.deadhead_router.nx")
    def test_create_deadhead_shapes_multiple_trips(self, mock_nx, mock_ox):
        # Test with multiple trips (serial processing to avoid pickling issues)
        mock_graph = MagicMock()
        mock_graph.nodes = {1: {"x": -105.0, "y": 39.0}, 2: {"x": -104.9, "y": 39.1}}
        mock_graph.get_edge_data.return_value = {0: {"length": 1000}}

        self.router.graph = mock_graph
        mock_ox.nearest_nodes.side_effect = [1, 2, 1, 2]  # Two trips
        mock_nx.shortest_path.side_effect = [[1, 2], [1, 2]]

        df = gpd.GeoDataFrame(
            {
                "block_id": ["B1", "B2"],
                "geometry_origin": [Point(-105.0, 39.0), Point(-105.0, 39.0)],
                "geometry_destination": [Point(-104.9, 39.1), Point(-104.9, 39.1)],
            }
        )

        # Test with serial processing (n_processes=1)
        out_df = self.router.create_deadhead_shapes(df, n_processes=1)

        self.assertIsInstance(out_df, pd.DataFrame)
        self.assertTrue(len(out_df) >= 4)  # At least 2 points per trip
        unique_shapes = out_df["shape_id"].unique()
        self.assertEqual(len(unique_shapes), 2)


if __name__ == "__main__":
    unittest.main()
