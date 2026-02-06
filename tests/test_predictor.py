import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from pathlib import Path
from routee.transit.predictor import GTFSEnergyPredictor


class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.gtfs_path = "/tmp/fake_gtfs"
        self.predictor = GTFSEnergyPredictor(self.gtfs_path)

    def test_init(self):
        self.assertEqual(self.predictor.gtfs_path, Path(self.gtfs_path))
        self.assertTrue(self.predictor.trips.empty)
        self.assertTrue(self.predictor.shapes.empty)

    @patch("routee.transit.predictor.Feed.from_dir")
    def test_load_gtfs_data(self, mock_from_dir):
        # Mock Feed
        mock_feed = MagicMock()

        # Mock agency dataframe
        mock_feed.agency = pd.DataFrame({"agency_name": ["Agency1"]})

        # Mock trips dataframe
        mock_feed.trips = pd.DataFrame({"service_id": ["S1"]})

        mock_feed.get_trips_from_sids.return_value = pd.DataFrame(
            {"trip_id": ["T1"], "service_id": ["S1"], "shape_id": ["SH1"]}
        )

        # Mock shapes dataframe
        mock_feed.shapes = pd.DataFrame({"shape_id": ["SH1"]})

        mock_from_dir.return_value = mock_feed

        self.predictor.load_gtfs_data()

        self.assertEqual(len(self.predictor.trips), 1)
        self.assertEqual(self.predictor.trips.iloc[0]["trip_id"], "T1")
        self.assertEqual(len(self.predictor.shapes), 1)

    @patch("routee.transit.predictor.filter_blocks_by_route")
    def test_filter_trips(self, mock_filter):
        # Setup pre-loaded state
        self.predictor.feed = MagicMock()
        self.predictor.feed.get_service_ids_from_date.return_value = ["S1"]
        self.predictor.trips = pd.DataFrame(
            {
                "trip_id": ["T1", "T2"],
                "service_id": ["S1", "S2"],
                "shape_id": ["SH1", "SH2"],
            }
        )
        self.predictor.feed.shapes = pd.DataFrame({"shape_id": ["SH1", "SH2"]})

        # Test filtering by date
        self.predictor.filter_trips(date="2023-01-01")
        self.assertEqual(len(self.predictor.trips), 1)
        self.assertEqual(self.predictor.trips.iloc[0]["service_id"], "S1")

    def test_add_trip_times(self):
        # Setup pre-loaded state
        self.predictor.feed = MagicMock()
        self.predictor.feed.stop_times = pd.DataFrame(
            {
                "trip_id": ["T1", "T1"],
                "arrival_time": [
                    pd.Timedelta(hours=8),
                    pd.Timedelta(hours=8, minutes=30),
                ],
            }
        )
        self.predictor.trips = pd.DataFrame({"trip_id": ["T1"], "service_id": ["S1"]})
        # Mock service ID counts
        self.predictor.feed.get_service_ids_all_dates.return_value = pd.DataFrame(
            {"service_id": ["S1"], "date": ["2023-01-01"]}
        )

        self.predictor.add_trip_times()

        self.assertIn("start_time", self.predictor.trips.columns)
        self.assertIn("trip_duration_minutes", self.predictor.trips.columns)
        self.assertEqual(self.predictor.trips.iloc[0]["trip_duration_minutes"], 30.0)


if __name__ == "__main__":
    unittest.main()
