import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from routee.transit.gtfs_processing import (
    upsample_shape,
    add_stop_flags_to_shape,
    estimate_trip_timestamps,
    extend_trip_traces,
)


class TestGTFSProcessing(unittest.TestCase):
    def test_upsample_shape(self) -> None:
        # Create a simple shape with 2 points
        df = pd.DataFrame(
            {
                "shape_pt_lat": [40.0, 40.1],
                "shape_pt_lon": [-105.0, -105.1],
                "shape_id": ["S1", "S1"],
            }
        )

        upsampled = upsample_shape(df)

        # In upsample_shape, the timestamp is used as index then reset_index(drop=True) is called
        # so it's not in the columns anymore, but it's used for resampling.
        # Let's verify shape_dist_traveled is present and interpolated.
        self.assertIn("shape_dist_traveled", upsampled.columns)
        self.assertEqual(upsampled["shape_id"].iloc[0], "S1")
        # Ensure it's roughly 1Hz (depends on distance and 30km/h assumption)
        # Distance is roughly 14km, at 30km/h (8.33 m/s) it should be around 1700 samples
        self.assertGreater(len(upsampled), 1000)

    def test_add_stop_flags_to_shape(self) -> None:
        trip_shape_df = pd.DataFrame(
            {
                "trip_id": ["T1", "T1", "T1"],
                "shape_pt_lon": [-105.0, -105.05, -105.1],
                "shape_pt_lat": [40.0, 40.05, 40.1],
                "coordinate_id": [0, 1, 2],
            }
        )

        stop_times_ext = pd.DataFrame(
            {
                "trip_id": ["T1", "T1"],
                "stop_lon": [-105.0, -105.1],
                "stop_lat": [40.0, 40.1],
            }
        )

        result = add_stop_flags_to_shape(trip_shape_df, stop_times_ext)

        self.assertIn("with_stop", result.columns)
        self.assertEqual(result.iloc[0]["with_stop"], 1)
        self.assertEqual(result.iloc[1]["with_stop"], 0)
        self.assertEqual(result.iloc[2]["with_stop"], 1)

    def test_estimate_trip_timestamps(self) -> None:
        df = pd.DataFrame(
            {
                "shape_dist_traveled": [0, 500, 1000],
                "start_time": ["08:00:00", "08:00:00", "08:00:00"],
                "end_time": ["08:10:00", "08:10:00", "08:10:00"],
            }
        )

        result = estimate_trip_timestamps(df)

        self.assertIn("timestamp", result.columns)
        self.assertIn("hour", result.columns)
        self.assertIn("minute", result.columns)

        # 50% distance should be 50% time (08:05:00)
        self.assertEqual(result.iloc[1]["timestamp"], pd.Timedelta(hours=8, minutes=5))

    @patch("routee.transit.gtfs_processing.mp.Pool")
    def test_extend_trip_traces(self, mock_pool_cls: MagicMock) -> None:
        # Setup mock Feed
        mock_feed = MagicMock()
        mock_feed.stop_times = pd.DataFrame(
            {"trip_id": ["T1"], "stop_sequence": [1], "stop_id": ["S1"]}
        )
        mock_feed.stops = pd.DataFrame(
            {"stop_id": ["S1"], "stop_lat": [40.0], "stop_lon": [-105.0]}
        )

        trips_df = pd.DataFrame(
            {
                "trip_id": ["T1"],
                "shape_id": ["SH1"],
                "start_time": ["08:00:00"],
                "end_time": ["08:10:00"],
            }
        )

        matched_shapes_df = pd.DataFrame(
            {"shape_id": ["SH1", "SH1"], "shape_dist_traveled": [0, 1000]}
        )

        # Mocking the pool.map to just return the input list
        mock_pool = MagicMock()
        mock_pool_cls.return_value.__enter__.return_value = mock_pool
        mock_pool.map.side_effect = lambda func, items: [func(item) for item in items]

        result = extend_trip_traces(
            trips_df, matched_shapes_df, mock_feed, n_processes=1
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("timestamp", result.columns)
        self.assertEqual(result.iloc[0]["trip_id"], "T1")


if __name__ == "__main__":
    unittest.main()
