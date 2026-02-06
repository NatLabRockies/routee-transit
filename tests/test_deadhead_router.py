import unittest
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from routee.transit.deadhead_router import (
    create_deadhead_shapes,
    route_single_trip_fallback,
    _haversine_km,
)


class TestDeadheadRouter(unittest.TestCase):
    def setUp(self) -> None:
        self.app = MagicMock()

    def test_haversine_km(self) -> None:
        # Distance between (0,0) and (0,1) lat is approx 111.19 km
        dist = _haversine_km(0, 0, 1, 0)
        self.assertAlmostEqual(dist, 111.19, places=1)

    def test_haversine_km_edge_cases(self) -> None:
        # Same point should return 0
        dist = _haversine_km(39.0, -105.0, 39.0, -105.0)
        self.assertAlmostEqual(dist, 0.0, places=5)

        # Equator to equator (1 degree lon) is approx 111.19 km
        dist = _haversine_km(0, 0, 0, 1)
        self.assertAlmostEqual(dist, 111.19, places=1)

    def test_route_single_trip_no_route_fallback(self) -> None:
        # Test fallback when no edges are found
        rows = route_single_trip_fallback(-105.0, 39.0, -104.9, 39.1, "B1")

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["shape_pt_lon"], -105.0)
        self.assertEqual(rows[-1]["shape_pt_lon"], -104.9)
        self.assertGreater(rows[-1]["shape_dist_traveled"], 0)

    @patch("routee.transit.deadhead_router.geometry_from_route")
    def test_create_deadhead_shapes(self, mock_geom_from_route: MagicMock) -> None:
        from shapely.geometry import LineString

        # Setup mock app
        self.app.run.return_value = [{"route": {"path": "mock_path"}}]
        mock_geom_from_route.return_value = LineString([(-105.0, 39.0), (-104.9, 39.1)])

        df = gpd.GeoDataFrame(
            {
                "block_id": ["B1"],
                "geometry_origin": [Point(-105.0, 39.0)],
                "geometry_destination": [Point(-104.9, 39.1)],
            }
        )

        out_df = create_deadhead_shapes(self.app, df)

        self.assertIsInstance(out_df, pd.DataFrame)
        self.assertTrue(len(out_df) >= 2)
        self.assertEqual(out_df.iloc[0]["shape_id"], "B1")
        self.assertEqual(out_df.iloc[0]["shape_pt_sequence"], 1)

    @patch("routee.transit.deadhead_router.geometry_from_route")
    def test_route_single_trip_with_geometry(
        self, mock_geom_from_route: MagicMock
    ) -> None:
        from shapely.geometry import LineString

        # Test with result that has geometry
        geom = LineString([(-105.0, 39.0), (-104.95, 39.05), (-104.9, 39.1)])
        mock_geom_from_route.return_value = geom

        self.app.run.return_value = [{"route": {"path": "mock_path"}}]

        df = gpd.GeoDataFrame(
            {
                "block_id": ["B1"],
                "geometry_origin": [Point(-105.0, 39.0)],
                "geometry_destination": [Point(-104.9, 39.1)],
            }
        )
        out_df = create_deadhead_shapes(self.app, df)

        self.assertEqual(len(out_df), 3)  # Should have 3 points from the LineString
        self.assertEqual(out_df.iloc[0]["shape_id"], "B1")
        self.assertEqual(out_df.iloc[0]["shape_pt_sequence"], 1)
        self.assertEqual(out_df.iloc[0]["shape_dist_traveled"], 0.0)
        # Last point should have accumulated distance
        self.assertGreater(out_df.iloc[-1]["shape_dist_traveled"], 0)

        # Verify that it used the geometry we provided
        self.assertEqual(out_df.iloc[1]["shape_pt_lon"], -104.95)

    @patch("routee.transit.deadhead_router.geometry_from_route")
    def test_create_deadhead_shapes_multiple_trips(
        self, mock_geom_from_route: MagicMock
    ) -> None:
        from shapely.geometry import LineString

        # Test with multiple trips
        self.app.run.return_value = [
            {"route": {"path": "mock_path_1"}},
            {"route": {"path": "mock_path_2"}},
        ]
        mock_geom_from_route.side_effect = [
            LineString([(-105.0, 39.0), (-104.9, 39.1)]),
            LineString([(-105.0, 39.0), (-104.9, 39.1)]),
        ]

        df = gpd.GeoDataFrame(
            {
                "block_id": ["B1", "B2"],
                "geometry_origin": [Point(-105.0, 39.0), Point(-105.0, 39.0)],
                "geometry_destination": [Point(-104.9, 39.1), Point(-104.9, 39.1)],
            }
        )

        out_df = create_deadhead_shapes(self.app, df)

        self.assertIsInstance(out_df, pd.DataFrame)
        self.assertTrue(len(out_df) >= 4)  # At least 2 points per trip
        unique_shapes = out_df["shape_id"].unique()
        self.assertEqual(len(unique_shapes), 2)

    @patch("routee.transit.deadhead_router.log")
    def test_create_deadhead_shapes_compass_error_fallback(
        self, mock_log: MagicMock
    ) -> None:
        # Test that a compass error triggers a fallback and a warning log
        self.app.run.return_value = [{"error": "Something went wrong"}]

        df = gpd.GeoDataFrame(
            {
                "block_id": ["B1"],
                "geometry_origin": [Point(-105.0, 39.0)],
                "geometry_destination": [Point(-104.9, 39.1)],
            }
        )

        out_df = create_deadhead_shapes(self.app, df)

        self.assertEqual(len(out_df), 2)  # Straight line fallback
        mock_log.warning.assert_called_once()
        self.assertIn(
            "CompassApp failed for block_id B1", mock_log.warning.call_args[0][0]
        )

    @patch("routee.transit.deadhead_router.geometry_from_route")
    def test_create_deadhead_shapes_geometry_parsing_failure(
        self, mock_geom_from_route: MagicMock
    ) -> None:
        # Test that geometry parsing failure raises an error
        self.app.run.return_value = [{"route": {"path": "corrupt"}}]
        mock_geom_from_route.side_effect = ValueError("Parsing failed")

        df = gpd.GeoDataFrame(
            {
                "block_id": ["B1"],
                "geometry_origin": [Point(-105.0, 39.0)],
                "geometry_destination": [Point(-104.9, 39.1)],
            }
        )

        with self.assertRaises(ValueError) as cm:
            create_deadhead_shapes(self.app, df)
        self.assertEqual(str(cm.exception), "Parsing failed")
