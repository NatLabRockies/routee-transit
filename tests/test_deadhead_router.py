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
        self.assertIsNone(self.router.app)

    def test_from_geometries(self):
        geoms = pd.Series([Point(-105.0, 39.0), Point(-104.8, 39.2)])
        router = NetworkRouter.from_geometries(
            geoms, buffer_deg_lat=0.1, buffer_deg_lon=0.1
        )

        # Expected bbox: (-105.1, 38.9, -104.7, 39.3)
        expected_bbox = (-105.1, 38.9, -104.7, 39.3)
        for act, exp in zip(router.bbox, expected_bbox):
            self.assertAlmostEqual(act, exp)

    @patch("osmnx.graph_from_bbox")
    @patch("nrel.routee.transit.deadhead_router.CompassApp")
    def test_ensure_app_loaded(self, mock_compass, mock_ox_bbox):
        mock_graph = MagicMock()
        mock_ox_bbox.return_value = mock_graph
        mock_app = MagicMock()
        mock_compass.from_graph.return_value = mock_app

        self.router._ensure_app_loaded()

        mock_ox_bbox.assert_called_once_with(self.bbox, network_type="drive")
        mock_compass.from_graph.assert_called_once_with(mock_graph)
        self.assertEqual(self.router.app, mock_app)

    @patch("nrel.routee.transit.deadhead_router.geometry_from_route")
    def test_create_deadhead_shapes(self, mock_geom_from_route):
        from shapely.geometry import LineString

        # Setup mock app
        mock_app = MagicMock()
        mock_app.run.return_value = [{"route": {"path": "mock_path"}}]
        mock_geom_from_route.return_value = LineString([(-105.0, 39.0), (-104.9, 39.1)])

        self.router.app = mock_app

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

    def test_route_single_trip_no_route_fallback(self):
        # Test fallback when no edges are found
        rows = self.router._route_single_trip_fallback(-105.0, 39.0, -104.9, 39.1, "B1")

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

    @patch("nrel.routee.transit.deadhead_router.geometry_from_route")
    def test_route_single_trip_with_geometry(self, mock_geom_from_route):
        from shapely.geometry import LineString

        # Test with result that has geometry
        geom = LineString([(-105.0, 39.0), (-104.95, 39.05), (-104.9, 39.1)])
        mock_geom_from_route.return_value = geom

        mock_app = MagicMock()
        mock_app.run.return_value = [{"route": {"path": "mock_path"}}]
        self.router.app = mock_app

        df = gpd.GeoDataFrame(
            {
                "block_id": ["B1"],
                "geometry_origin": [Point(-105.0, 39.0)],
                "geometry_destination": [Point(-104.9, 39.1)],
            }
        )
        out_df = self.router.create_deadhead_shapes(df)

        self.assertEqual(len(out_df), 3)  # Should have 3 points from the LineString
        self.assertEqual(out_df.iloc[0]["shape_id"], "B1")
        self.assertEqual(out_df.iloc[0]["shape_pt_sequence"], 1)
        self.assertEqual(out_df.iloc[0]["shape_dist_traveled"], 0.0)
        # Last point should have accumulated distance
        self.assertGreater(out_df.iloc[-1]["shape_dist_traveled"], 0)

        # Verify that it used the geometry we provided
        self.assertEqual(out_df.iloc[1]["shape_pt_lon"], -104.95)

    @patch("nrel.routee.transit.deadhead_router.geometry_from_route")
    def test_create_deadhead_shapes_multiple_trips(self, mock_geom_from_route):
        from shapely.geometry import LineString

        # Test with multiple trips
        mock_app = MagicMock()
        mock_app.run.return_value = [
            {"route": {"path": "mock_path_1"}},
            {"route": {"path": "mock_path_2"}},
        ]
        mock_geom_from_route.side_effect = [
            LineString([(-105.0, 39.0), (-104.9, 39.1)]),
            LineString([(-105.0, 39.0), (-104.9, 39.1)]),
        ]

        self.router.app = mock_app

        df = gpd.GeoDataFrame(
            {
                "block_id": ["B1", "B2"],
                "geometry_origin": [Point(-105.0, 39.0), Point(-105.0, 39.0)],
                "geometry_destination": [Point(-104.9, 39.1), Point(-104.9, 39.1)],
            }
        )

        out_df = self.router.create_deadhead_shapes(df, n_processes=1)

        self.assertIsInstance(out_df, pd.DataFrame)
        self.assertTrue(len(out_df) >= 4)  # At least 2 points per trip
        unique_shapes = out_df["shape_id"].unique()
        self.assertEqual(len(unique_shapes), 2)

    @patch("nrel.routee.transit.deadhead_router.log")
    def test_create_deadhead_shapes_compass_error_fallback(self, mock_log):
        # Test that a compass error triggers a fallback and a warning log
        mock_app = MagicMock()
        mock_app.run.return_value = [{"error": "Something went wrong"}]
        self.router.app = mock_app

        df = gpd.GeoDataFrame(
            {
                "block_id": ["B1"],
                "geometry_origin": [Point(-105.0, 39.0)],
                "geometry_destination": [Point(-104.9, 39.1)],
            }
        )

        out_df = self.router.create_deadhead_shapes(df)

        self.assertEqual(len(out_df), 2)  # Straight line fallback
        mock_log.warning.assert_called_once()
        self.assertIn(
            "CompassApp failed for block_id B1", mock_log.warning.call_args[0][0]
        )

    @patch("nrel.routee.transit.deadhead_router.geometry_from_route")
    def test_create_deadhead_shapes_geometry_parsing_failure(
        self, mock_geom_from_route
    ):
        # Test that geometry parsing failure raises an error
        mock_app = MagicMock()
        mock_app.run.return_value = [{"route": {"path": "corrupt"}}]
        mock_geom_from_route.side_effect = ValueError("Parsing failed")

        self.router.app = mock_app

        df = gpd.GeoDataFrame(
            {
                "block_id": ["B1"],
                "geometry_origin": [Point(-105.0, 39.0)],
                "geometry_destination": [Point(-104.9, 39.1)],
            }
        )

        with self.assertRaises(ValueError) as cm:
            self.router.create_deadhead_shapes(df)
        self.assertEqual(str(cm.exception), "Parsing failed")
