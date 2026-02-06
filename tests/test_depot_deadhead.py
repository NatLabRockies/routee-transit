import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from routee.transit.depot_deadhead import (
    create_depot_deadhead_stops,
    create_depot_deadhead_trips,
    get_default_depot_path,
    infer_depot_trip_endpoints,
)


class TestGetDefaultDepotPath(unittest.TestCase):
    def test_get_default_depot_path(self):
        path = get_default_depot_path()
        self.assertIsInstance(path, Path)
        self.assertTrue(str(path).endswith("FTA_Depot"))


class TestCreateDepotDeadheadTrips(unittest.TestCase):
    def setUp(self):
        # Create sample trips and stop_times dataframes
        self.trips_df = pd.DataFrame(
            {
                "trip_id": ["trip1", "trip2", "trip3", "trip4"],
                "block_id": ["block1", "block1", "block2", "block2"],
                "service_id": ["service1", "service1", "service2", "service2"],
                "agency_id": ["agency1", "agency1", "agency2", "agency2"],
            }
        )

        self.stop_times_df = pd.DataFrame(
            {
                "trip_id": ["trip1", "trip1", "trip2", "trip2", "trip3", "trip4"],
                "arrival_time": [
                    pd.Timedelta(hours=8),
                    pd.Timedelta(hours=8, minutes=30),
                    pd.Timedelta(hours=9),
                    pd.Timedelta(hours=9, minutes=30),
                    pd.Timedelta(hours=10),
                    pd.Timedelta(hours=11),
                ],
            }
        )

    def test_create_depot_deadhead_trips_basic(self):
        result = create_depot_deadhead_trips(self.trips_df, self.stop_times_df)

        # Should create 2 trips per block (pull-out and pull-in)
        self.assertEqual(len(result), 4)

        # Check pull-out trips
        pullout_trips = result[result["trip_type"] == "pull-out"]
        self.assertEqual(len(pullout_trips), 2)
        self.assertTrue(all(pullout_trips["trip_id"].str.startswith("depot_to_")))

        # Check pull-in trips
        pullin_trips = result[result["trip_type"] == "pull-in"]
        self.assertEqual(len(pullin_trips), 2)
        self.assertTrue(all(pullin_trips["trip_id"].str.endswith("_to_depot")))

    def test_create_depot_deadhead_trips_columns(self):
        result = create_depot_deadhead_trips(self.trips_df, self.stop_times_df)

        required_columns = [
            "trip_id",
            "trip_type",
            "route_id",
            "service_id",
            "block_id",
            "shape_id",
            "route_short_name",
            "route_type",
            "route_desc",
            "agency_id",
        ]

        for col in required_columns:
            self.assertIn(col, result.columns)

    def test_create_depot_deadhead_trips_route_type(self):
        result = create_depot_deadhead_trips(self.trips_df, self.stop_times_df)

        # All route types should be 3 (bus)
        self.assertTrue(all(result["route_type"] == 3))

    def test_create_depot_deadhead_trips_with_existing_deadhead(self):
        # Add a between-trip deadhead to trips
        trips_with_deadhead = self.trips_df.copy()
        trips_with_deadhead["from_trip"] = [None, "trip1", None, None]

        result = create_depot_deadhead_trips(trips_with_deadhead, self.stop_times_df)

        # Should still create 4 depot deadhead trips (2 per block)
        self.assertEqual(len(result), 4)


class TestInferDepotTripEndpoints(unittest.TestCase):
    def setUp(self):
        # Create sample trips
        self.trips_df = pd.DataFrame(
            {
                "trip_id": ["trip1", "trip2"],
                "block_id": ["block1", "block1"],
            }
        )

        # Create mock feed
        self.feed = MagicMock()
        self.feed.stop_times = pd.DataFrame(
            {
                "trip_id": ["trip1", "trip1", "trip2", "trip2"],
                "stop_id": ["stop1", "stop2", "stop3", "stop4"],
                "arrival_time": [
                    pd.Timedelta(hours=8),
                    pd.Timedelta(hours=8, minutes=30),
                    pd.Timedelta(hours=9),
                    pd.Timedelta(hours=9, minutes=30),
                ],
            }
        )
        self.feed.stops = pd.DataFrame(
            {
                "stop_id": ["stop1", "stop2", "stop3", "stop4"],
                "stop_lat": [39.0, 39.1, 39.2, 39.3],
                "stop_lon": [-105.0, -105.1, -105.2, -105.3],
            }
        )

        # Create temporary depot file
        self.temp_dir = tempfile.mkdtemp()
        self.depot_path = os.path.join(self.temp_dir, "depots.geojson")

        depot_gdf = gpd.GeoDataFrame(
            {"geometry": [Point(-105.05, 39.05), Point(-105.25, 39.25)]},
            crs="EPSG:4326",
        )
        depot_gdf.to_file(self.depot_path, driver="GeoJSON")

    def tearDown(self):
        # Clean up temporary files
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_infer_depot_trip_endpoints_returns_geodataframes(self):
        first_stops, last_stops = infer_depot_trip_endpoints(
            self.trips_df, self.feed, self.depot_path
        )

        self.assertIsInstance(first_stops, gpd.GeoDataFrame)
        self.assertIsInstance(last_stops, gpd.GeoDataFrame)

    def test_infer_depot_trip_endpoints_columns(self):
        first_stops, last_stops = infer_depot_trip_endpoints(
            self.trips_df, self.feed, self.depot_path
        )

        # Check required columns
        for gdf in [first_stops, last_stops]:
            self.assertIn("block_id", gdf.columns)
            self.assertIn("geometry_origin", gdf.columns)
            self.assertIn("geometry_destination", gdf.columns)

    def test_infer_depot_trip_endpoints_geometry_types(self):
        first_stops, last_stops = infer_depot_trip_endpoints(
            self.trips_df, self.feed, self.depot_path
        )

        # All geometries should be Points
        for gdf in [first_stops, last_stops]:
            for geom in gdf["geometry_origin"]:
                self.assertEqual(geom.geom_type, "Point")
            for geom in gdf["geometry_destination"]:
                self.assertEqual(geom.geom_type, "Point")

    def test_infer_depot_trip_endpoints_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            infer_depot_trip_endpoints(
                self.trips_df, self.feed, "/nonexistent/path/to/depots.geojson"
            )

    def test_infer_depot_trip_endpoints_crs(self):
        first_stops, last_stops = infer_depot_trip_endpoints(
            self.trips_df, self.feed, self.depot_path
        )

        # Both should be in EPSG:4326
        self.assertEqual(first_stops.crs.to_string(), "EPSG:4326")
        self.assertEqual(last_stops.crs.to_string(), "EPSG:4326")


class TestCreateDepotDeadheadStops(unittest.TestCase):
    def setUp(self):
        # Create sample GeoDataFrames with proper geometry column
        self.first_stops_gdf = gpd.GeoDataFrame(
            {
                "block_id": ["block1"],
                "arrival_time": [pd.Timedelta(hours=8)],
                "geometry_origin": [Point(-105.0, 39.0)],
                "geometry_destination": [Point(-105.1, 39.1)],
            },
            geometry="geometry_destination",
            crs="EPSG:4326",
        )

        self.last_stops_gdf = gpd.GeoDataFrame(
            {
                "block_id": ["block1"],
                "departure_time": [pd.Timedelta(hours=17)],
                "geometry_origin": [Point(-105.2, 39.2)],
                "geometry_destination": [Point(-105.3, 39.3)],
            },
            geometry="geometry_origin",
            crs="EPSG:4326",
        )

        self.deadhead_trips = pd.DataFrame(
            {
                "trip_id": ["depot_to_trip1", "trip2_to_depot"],
                "trip_type": ["pull-out", "pull-in"],
                "block_id": ["block1", "block1"],
            }
        )

    def test_create_depot_deadhead_stops_returns_dataframes(self):
        stop_times, stops = create_depot_deadhead_stops(
            self.first_stops_gdf, self.last_stops_gdf, self.deadhead_trips
        )

        self.assertIsInstance(stop_times, pd.DataFrame)
        self.assertIsInstance(stops, pd.DataFrame)

    def test_create_depot_deadhead_stops_columns(self):
        stop_times, stops = create_depot_deadhead_stops(
            self.first_stops_gdf, self.last_stops_gdf, self.deadhead_trips
        )

        # Check stop_times columns
        required_stop_times_cols = [
            "trip_id",
            "stop_sequence",
            "arrival_time",
            "stop_id",
            "departure_time",
            "shape_dist_traveled",
        ]
        for col in required_stop_times_cols:
            self.assertIn(col, stop_times.columns)

        # Check stops columns
        required_stops_cols = ["stop_id", "stop_lat", "stop_lon"]
        for col in required_stops_cols:
            self.assertIn(col, stops.columns)

    def test_create_depot_deadhead_stops_count(self):
        stop_times, stops = create_depot_deadhead_stops(
            self.first_stops_gdf, self.last_stops_gdf, self.deadhead_trips
        )

        # Each deadhead trip should have 2 stops (origin and destination)
        # 2 trips * 2 stops = 4 total
        self.assertEqual(len(stop_times), 4)
        self.assertEqual(len(stops), 4)

    def test_create_depot_deadhead_stops_sequences(self):
        stop_times, stops = create_depot_deadhead_stops(
            self.first_stops_gdf, self.last_stops_gdf, self.deadhead_trips
        )

        # Check stop sequences are 1 and 2 for each trip
        for trip_id in self.deadhead_trips["trip_id"]:
            trip_stops = stop_times[stop_times["trip_id"] == trip_id]
            sequences = sorted(trip_stops["stop_sequence"].tolist())
            self.assertEqual(sequences, [1, 2])

    def test_create_depot_deadhead_stops_stop_ids(self):
        stop_times, stops = create_depot_deadhead_stops(
            self.first_stops_gdf, self.last_stops_gdf, self.deadhead_trips
        )

        # All stop IDs should start with "depot_deadhead_"
        for stop_id in stops["stop_id"]:
            self.assertTrue(stop_id.startswith("depot_deadhead_"))

    def test_create_depot_deadhead_stops_coordinates(self):
        stop_times, stops = create_depot_deadhead_stops(
            self.first_stops_gdf, self.last_stops_gdf, self.deadhead_trips
        )

        # Check that coordinates are valid
        self.assertTrue(all(stops["stop_lat"].between(-90, 90)))
        self.assertTrue(all(stops["stop_lon"].between(-180, 180)))


if __name__ == "__main__":
    unittest.main()
