import unittest
from unittest.mock import MagicMock

import pandas as pd

from routee.transit.mid_block_deadhead import (
    create_mid_block_deadhead_stops,
    create_mid_block_deadhead_trips,
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
        # Route ID encodes last stop of from_trip → first stop of to_trip
        # T1 last stop is S2, T2 first stop is S3
        self.assertEqual(deadhead_trips.iloc[0]["route_id"], "deadhead_S2_to_S3")
        # route_short_name mirrors route_id
        self.assertEqual(
            deadhead_trips.iloc[0]["route_short_name"], "deadhead_S2_to_S3"
        )

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

        # Mid-block endpoints are existing GTFS stops — no new stops added
        self.assertEqual(len(stops), 0)

        # Verify ODs
        self.assertEqual(len(ods), 1)
        self.assertEqual(ods.iloc[0]["block_id"], "T1_to_T2")


class TestMidBlockDeadheadServiceIdGrouping(unittest.TestCase):
    """A single block ID maps to different trips depending on the service ID.

    Grouping by both ``service_id`` and ``block_id`` must keep the two services
    separate so deadhead trips are never created across service boundaries.
    """

    def setUp(self) -> None:
        # Block B1 is reused by two services with entirely different trips.
        # S1: T1 -> T2, S2: T3 -> T4. All share block_id B1.
        self.trips_df = pd.DataFrame(
            {
                "trip_id": ["T1", "T2", "T3", "T4"],
                "route_id": ["R1", "R1", "R2", "R2"],
                "service_id": ["S1", "S1", "S2", "S2"],
                "block_id": ["B1", "B1", "B1", "B1"],
                "shape_id": ["SH1", "SH2", "SH3", "SH4"],
            }
        )

        self.stop_times_df = pd.DataFrame(
            {
                "trip_id": ["T1", "T1", "T2", "T2", "T3", "T3", "T4", "T4"],
                "arrival_time": [
                    pd.Timedelta(hours=8),
                    pd.Timedelta(hours=8, minutes=30),
                    pd.Timedelta(hours=9),
                    pd.Timedelta(hours=9, minutes=30),
                    pd.Timedelta(hours=8),
                    pd.Timedelta(hours=8, minutes=30),
                    pd.Timedelta(hours=9),
                    pd.Timedelta(hours=9, minutes=30),
                ],
                "departure_time": [
                    pd.Timedelta(hours=8),
                    pd.Timedelta(hours=8, minutes=30),
                    pd.Timedelta(hours=9),
                    pd.Timedelta(hours=9, minutes=30),
                    pd.Timedelta(hours=8),
                    pd.Timedelta(hours=8, minutes=30),
                    pd.Timedelta(hours=9),
                    pd.Timedelta(hours=9, minutes=30),
                ],
                "stop_id": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"],
                "stop_sequence": [1, 2, 1, 2, 1, 2, 1, 2],
            }
        )

    def test_deadhead_trips_are_grouped_by_service_and_block(self) -> None:
        deadhead_trips = create_mid_block_deadhead_trips(
            self.trips_df, self.stop_times_df
        )

        # Exactly one deadhead per service: T1->T2 (S1) and T3->T4 (S2).
        # Grouping by block_id alone would incorrectly connect trips across the
        # two services (e.g. T1->T3 or T2->T3).
        self.assertEqual(len(deadhead_trips), 2)

        trip_ids = set(deadhead_trips["trip_id"])
        self.assertEqual(trip_ids, {"T1_to_T2", "T3_to_T4"})

        # No cross-service deadhead trips should exist.
        self.assertNotIn("T2_to_T3", trip_ids)
        self.assertNotIn("T1_to_T3", trip_ids)

        by_trip = deadhead_trips.set_index("trip_id")
        # S1 deadhead: T1 last stop S2 -> T2 first stop S3
        self.assertEqual(by_trip.loc["T1_to_T2", "service_id"], "S1")
        self.assertEqual(by_trip.loc["T1_to_T2", "route_id"], "deadhead_S2_to_S3")
        # S2 deadhead: T3 last stop S6 -> T4 first stop S7
        self.assertEqual(by_trip.loc["T3_to_T4", "service_id"], "S2")
        self.assertEqual(by_trip.loc["T3_to_T4", "route_id"], "deadhead_S6_to_S7")


class TestMidBlockDeadheadAgencyId(unittest.TestCase):
    """agency_id is inherited from the preceding (from) trip when present."""

    def setUp(self) -> None:
        # T1 -> T2 on one block; T1 and T2 carry different agencies so the test
        # can confirm the deadhead inherits the from-trip's (T1) agency.
        self.trips_df = pd.DataFrame(
            {
                "trip_id": ["T1", "T2"],
                "route_id": ["R1", "R1"],
                "service_id": ["S1", "S1"],
                "block_id": ["B1", "B1"],
                "shape_id": ["SH1", "SH2"],
                "agency_id": ["A1", "A2"],
            }
        )

        self.stop_times_df = pd.DataFrame(
            {
                "trip_id": ["T1", "T1", "T2", "T2"],
                "arrival_time": [
                    pd.Timedelta(hours=8),
                    pd.Timedelta(hours=8, minutes=30),
                    pd.Timedelta(hours=9),
                    pd.Timedelta(hours=9, minutes=30),
                ],
                "departure_time": [
                    pd.Timedelta(hours=8),
                    pd.Timedelta(hours=8, minutes=30),
                    pd.Timedelta(hours=9),
                    pd.Timedelta(hours=9, minutes=30),
                ],
                "stop_id": ["S1", "S2", "S3", "S4"],
                "stop_sequence": [1, 2, 1, 2],
            }
        )

    def test_agency_id_inherited_from_from_trip(self) -> None:
        deadhead_trips = create_mid_block_deadhead_trips(
            self.trips_df, self.stop_times_df
        )

        self.assertEqual(len(deadhead_trips), 1)
        self.assertIn("agency_id", deadhead_trips.columns)
        # Deadhead T1_to_T2 inherits the from-trip (T1) agency, not T2's.
        self.assertEqual(deadhead_trips.iloc[0]["agency_id"], "A1")

    def test_agency_id_absent_when_not_provided(self) -> None:
        # Drop the agency_id column entirely to mimic a feed without agencies.
        trips_no_agency = self.trips_df.drop(columns=["agency_id"])

        deadhead_trips = create_mid_block_deadhead_trips(
            trips_no_agency, self.stop_times_df
        )

        # Trips are still created; no agency_id column is fabricated.
        self.assertEqual(len(deadhead_trips), 1)
        self.assertNotIn("agency_id", deadhead_trips.columns)


if __name__ == "__main__":
    unittest.main()
