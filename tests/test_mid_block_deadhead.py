import unittest
import pandas as pd
from unittest.mock import MagicMock
from routee.transit.mid_block_deadhead import (
    create_mid_block_deadhead_trips,
    create_mid_block_deadhead_stops,
)


class TestMidBlockDeadhead(unittest.TestCase):
    def setUp(self) -> None:
        # Sample trips on two blocks
        self.trips_df = pd.DataFrame(
            {
                "trip_id": ["T1", "T2", "T3"],
                "route_id": ["R1", "R1", "R1"],
                "service_id": ["S1", "S1", "S1"],
                "block_id": ["B1", "B1", "B2"],
                "shape_id": ["SH1", "SH2", "SH3"],
            }
        )

        # Sample stop times - Trip 1 and Trip 2 are on the same block
        # Trip 1: 08:00 to 08:30
        # Trip 2: 09:00 to 09:30 (Gap from 08:30 to 09:00)
        self.stop_times_df = pd.DataFrame(
            {
                "trip_id": ["T1", "T1", "T2", "T2", "T3", "T3"],
                "arrival_time": [
                    pd.Timedelta(hours=8),
                    pd.Timedelta(hours=8, minutes=30),
                    pd.Timedelta(hours=9),
                    pd.Timedelta(hours=9, minutes=30),
                    pd.Timedelta(hours=10),
                    pd.Timedelta(hours=10, minutes=30),
                ],
                "departure_time": [
                    pd.Timedelta(hours=8),
                    pd.Timedelta(hours=8, minutes=30),
                    pd.Timedelta(hours=9),
                    pd.Timedelta(hours=9, minutes=30),
                    pd.Timedelta(hours=10),
                    pd.Timedelta(hours=10, minutes=30),
                ],
                "stop_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
                "stop_sequence": [1, 2, 1, 2, 1, 2],
            }
        )

    def test_create_mid_block_deadhead_trips(self) -> None:
        # Should find one deadhead trip: T1 -> T2
        deadhead_trips = create_mid_block_deadhead_trips(
            self.trips_df, self.stop_times_df
        )

        self.assertEqual(len(deadhead_trips), 1)
        self.assertEqual(deadhead_trips.iloc[0]["trip_id"], "T1_to_T2")
        self.assertEqual(deadhead_trips.iloc[0]["trip_type"], "mid_block_deadhead")
        self.assertEqual(deadhead_trips.iloc[0]["block_id"], "B1")

    def test_create_mid_block_deadhead_stops(self) -> None:
        deadhead_trips = create_mid_block_deadhead_trips(
            self.trips_df, self.stop_times_df
        )

        # Mock Feed
        mock_feed = MagicMock()
        mock_feed.stop_times = self.stop_times_df
        mock_feed.stops = pd.DataFrame(
            {
                "stop_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
                "stop_lat": [40.0, 40.1, 40.2, 40.3, 40.4, 40.5],
                "stop_lon": [-105.0, -105.1, -105.2, -105.3, -105.4, -105.5],
            }
        )

        stop_times, stops, ods = create_mid_block_deadhead_stops(
            mock_feed, deadhead_trips
        )

        # 1 deadhead trip -> 2 stop entries
        self.assertEqual(len(stop_times), 2)
        self.assertEqual(stop_times.iloc[0]["trip_id"], "T1_to_T2")

        # Verify locations in stops df
        # T1 last stop is S2 (40.1, -105.1)
        # T2 first stop is S3 (40.2, -105.2)
        self.assertIn(40.1, stops["stop_lat"].values)
        self.assertIn(40.2, stops["stop_lat"].values)

        # Verify ODs
        self.assertEqual(len(ods), 1)
        self.assertEqual(ods.iloc[0]["block_id"], "T1_to_T2")


if __name__ == "__main__":
    unittest.main()
