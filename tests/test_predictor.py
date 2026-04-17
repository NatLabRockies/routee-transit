import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from pathlib import Path
from routee.transit.predictor import (
    GGE_PER_GALLON_DIESEL,
    KWH_PER_GGE,
    GTFSEnergyPredictor,
)


class TestPredictor(unittest.TestCase):
    def setUp(self) -> None:
        self.gtfs_path = "/tmp/fake_gtfs"
        self.predictor = GTFSEnergyPredictor(self.gtfs_path)

    def test_init(self) -> None:
        self.assertEqual(self.predictor.gtfs_path, Path(self.gtfs_path))
        self.assertTrue(self.predictor.trips.empty)
        self.assertTrue(self.predictor.shapes.empty)

    @patch("routee.transit.predictor.Feed.from_dir")
    def test_load_gtfs_data(self, mock_from_dir: MagicMock) -> None:
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
    def test_filter_trips(self, mock_filter: MagicMock) -> None:
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

    def test_add_trip_times(self) -> None:
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


class TestPredictEnergyMPGe(unittest.TestCase):
    """Tests for predict_energy() focusing on MPGe calculation and column handling."""

    def setUp(self) -> None:
        self.predictor = GTFSEnergyPredictor("/tmp/fake_gtfs")
        # Minimal feed mock — predict_energy() checks `self.feed is None`
        self.predictor.feed = MagicMock()

        # Two trips sharing one shape so merge produces two output rows
        self.predictor.trips = pd.DataFrame(
            {
                "trip_id": ["T1", "T2"],
                "shape_id": ["SH1", "SH1"],
                "service_id": ["S1", "S1"],
                "route_short_name": ["101", "101"],
                "route_desc": ["Main Line", "Main Line"],
                "route_type": [3, 3],
            }
        )
        # Minimal shapes DataFrame (two points for one shape)
        self.predictor.shapes = pd.DataFrame(
            {
                "shape_id": ["SH1", "SH1"],
                "shape_pt_lat": [39.7, 39.71],
                "shape_pt_lon": [-104.99, -104.98],
                "shape_pt_sequence": [1, 2],
            }
        )
        # Pre-built mock app so load_compass_app() is never called
        self.mock_app = MagicMock()
        self.predictor.app = self.mock_app

        # Link results returned by the mocked _match_shapes_to_network
        self._link_results = pd.DataFrame({"shape_id": ["SH1"], "edge_id": [1001]})

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _make_compass_result(
        self, energy_value: float, distance_value: float, energy_field: str
    ) -> list[dict]:  # type: ignore[type-arg]
        return [
            {
                "route": {
                    "traversal_summary": {
                        energy_field: {"value": energy_value},
                        "edge_distance": {"value": distance_value},
                    }
                }
            }
        ]

    def _run_predict_energy(self, vehicle_models: list[str]) -> dict[str, pd.DataFrame]:
        """Run predict_energy() with multiprocessing and map-matching mocked out."""
        self.predictor.vehicle_models = vehicle_models

        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.map.return_value = [MagicMock()]  # upsampled shape groups

        with (
            patch("routee.transit.predictor.mp.Pool", return_value=mock_pool),
            patch.object(
                self.predictor,
                "_match_shapes_to_network",
                return_value=self._link_results,
            ),
        ):
            return self.predictor.predict_energy()

    # ------------------------------------------------------------------
    # MPGe calculation — BEB
    # ------------------------------------------------------------------

    def test_beb_mpge_calculation(self) -> None:
        """MPGe for a Battery Electric Bus equals miles / (kWh / KWH_PER_GGE)."""
        energy_kwh = KWH_PER_GGE  # exactly 1 GGE, so mpge == distance in miles
        distance_miles = 10.0
        self.mock_app.run_calculate_path.return_value = self._make_compass_result(
            energy_value=energy_kwh,
            distance_value=distance_miles,
            energy_field="trip_energy_electric",
        )

        results = self._run_predict_energy(["Transit_Bus_Battery_Electric"])
        trip_df = results["Transit_Bus_Battery_Electric_trip"]

        self.assertIn("mpge", trip_df.columns)
        # gge_per_unit = 1 / KWH_PER_GGE  →  gge_consumed = 1.0  →  mpge = 10.0
        expected = distance_miles / (energy_kwh / KWH_PER_GGE)
        for _, row in trip_df.iterrows():
            self.assertAlmostEqual(row["mpge"], expected, places=5)

    # ------------------------------------------------------------------
    # MPGe calculation — Diesel
    # ------------------------------------------------------------------

    def test_diesel_mpge_calculation(self) -> None:
        """MPGe for a Diesel Bus equals miles / (gallons * GGE_PER_GALLON_DIESEL)."""
        energy_gallons = 2.0
        distance_miles = 10.0
        self.mock_app.run_calculate_path.return_value = self._make_compass_result(
            energy_value=energy_gallons,
            distance_value=distance_miles,
            energy_field="trip_energy_liquid",
        )

        results = self._run_predict_energy(["Transit_Bus_Diesel"])
        trip_df = results["Transit_Bus_Diesel_trip"]

        self.assertIn("mpge", trip_df.columns)
        expected = distance_miles / (energy_gallons * GGE_PER_GALLON_DIESEL)
        for _, row in trip_df.iterrows():
            self.assertAlmostEqual(row["mpge"], expected, places=5)

    # ------------------------------------------------------------------
    # NaN handling: zero energy
    # ------------------------------------------------------------------

    def test_zero_energy_yields_nan_mpge(self) -> None:
        """Zero energy consumption must produce NaN (not inf) MPGe."""
        self.mock_app.run_calculate_path.return_value = self._make_compass_result(
            energy_value=0.0,
            distance_value=10.0,
            energy_field="trip_energy_electric",
        )

        results = self._run_predict_energy(["Transit_Bus_Battery_Electric"])
        trip_df = results["Transit_Bus_Battery_Electric_trip"]

        self.assertTrue(trip_df["mpge"].isna().all())

    # ------------------------------------------------------------------
    # NaN handling: negative energy
    # ------------------------------------------------------------------

    def test_negative_energy_yields_nan_mpge(self) -> None:
        """Negative energy consumption must produce NaN MPGe."""
        self.mock_app.run_calculate_path.return_value = self._make_compass_result(
            energy_value=-5.0,
            distance_value=10.0,
            energy_field="trip_energy_electric",
        )

        results = self._run_predict_energy(["Transit_Bus_Battery_Electric"])
        trip_df = results["Transit_Bus_Battery_Electric_trip"]

        self.assertTrue(trip_df["mpge"].isna().all())

    # ------------------------------------------------------------------
    # Column handling
    # ------------------------------------------------------------------

    def test_irrelevant_columns_dropped(self) -> None:
        """service_id, route_desc, route_type must not appear in results."""
        self.mock_app.run_calculate_path.return_value = self._make_compass_result(
            energy_value=KWH_PER_GGE,
            distance_value=10.0,
            energy_field="trip_energy_electric",
        )

        results = self._run_predict_energy(["Transit_Bus_Battery_Electric"])
        trip_df = results["Transit_Bus_Battery_Electric_trip"]

        for col in ["service_id", "route_desc", "route_type"]:
            self.assertNotIn(col, trip_df.columns, f"Column '{col}' should be dropped")

    def test_core_columns_preserved(self) -> None:
        """trip_id, energy_used, miles, mpge, and energy_unit must be present in results."""
        self.mock_app.run_calculate_path.return_value = self._make_compass_result(
            energy_value=KWH_PER_GGE,
            distance_value=10.0,
            energy_field="trip_energy_electric",
        )

        results = self._run_predict_energy(["Transit_Bus_Battery_Electric"])
        trip_df = results["Transit_Bus_Battery_Electric_trip"]

        for col in ["trip_id", "energy_used", "miles", "mpge", "energy_unit"]:
            self.assertIn(col, trip_df.columns, f"Column '{col}' should be present")


class TestGetAgencyOverview(unittest.TestCase):
    """Tests for the get_agency_overview() method."""

    def _make_predictor_with_predictions(
        self,
        trip_rows: list[dict],  # type: ignore[type-arg]
        agency_df: pd.DataFrame | None = None,
        total_service_days: int = 5,
    ) -> GTFSEnergyPredictor:
        predictor = GTFSEnergyPredictor("/tmp/fake_gtfs")

        # Inject trip-level energy predictions
        predictor.energy_predictions["trip"] = pd.DataFrame(trip_rows)

        # Optionally attach a mock feed
        if agency_df is not None:
            mock_feed = MagicMock()
            mock_feed.agency = agency_df
            # Return a DataFrame with one date row per service_id * day
            date_df = pd.DataFrame(
                {
                    "service_id": ["S1"] * total_service_days,
                    "date": pd.date_range("2023-01-02", periods=total_service_days),
                }
            )
            mock_feed.get_service_ids_all_dates.return_value = date_df
            predictor.feed = mock_feed

        return predictor

    # ------------------------------------------------------------------
    # Basic structure
    # ------------------------------------------------------------------

    def test_returns_dataframe_with_agency_rows(self) -> None:
        """One row per unique agency_id must be returned."""
        predictor = self._make_predictor_with_predictions(
            [
                {
                    "trip_id": "T1",
                    "trip_type": "service",
                    "agency_id": "A1",
                    "trip_count": 5,
                    "trip_duration_minutes": 30.0,
                    "miles": 10.0,
                },
                {
                    "trip_id": "T2",
                    "trip_type": "service",
                    "agency_id": "A2",
                    "trip_count": 3,
                    "trip_duration_minutes": 20.0,
                    "miles": 6.0,
                },
            ]
        )

        overview = predictor.get_agency_overview()

        self.assertEqual(len(overview), 2)
        self.assertIn("agency_id", overview.columns)

    def test_required_metric_columns_present(self) -> None:
        """trips_per_day, avg_trip_duration_minutes, avg_trip_distance_miles must exist."""
        predictor = self._make_predictor_with_predictions(
            [
                {
                    "trip_id": "T1",
                    "trip_type": "service",
                    "agency_id": "A1",
                    "trip_count": 10,
                    "trip_duration_minutes": 25.0,
                    "miles": 8.0,
                }
            ],
            total_service_days=10,
        )

        overview = predictor.get_agency_overview()

        for col in [
            "trips_per_day",
            "avg_trip_duration_minutes",
            "avg_trip_distance_miles",
        ]:
            self.assertIn(col, overview.columns, f"'{col}' should be in overview")

    # ------------------------------------------------------------------
    # trips_per_day calculation
    # ------------------------------------------------------------------

    def test_trips_per_day_calculation(self) -> None:
        """trips_per_day = sum(trip_count) / total_service_days."""
        # Provide a minimal agency_df so the mock feed is created and
        # get_service_ids_all_dates() returns the expected number of dates.
        agency_df = pd.DataFrame({"agency_id": ["A1"], "agency_name": ["Agency One"]})
        predictor = self._make_predictor_with_predictions(
            [
                {
                    "trip_id": "T1",
                    "trip_type": "service",
                    "agency_id": "A1",
                    "trip_count": 10,
                    "trip_duration_minutes": 30.0,
                    "miles": 10.0,
                },
                {
                    "trip_id": "T2",
                    "trip_type": "service",
                    "agency_id": "A1",
                    "trip_count": 5,
                    "trip_duration_minutes": 20.0,
                    "miles": 5.0,
                },
            ],
            agency_df=agency_df,
            total_service_days=5,
        )

        overview = predictor.get_agency_overview()

        row = overview[overview["agency_id"] == "A1"].iloc[0]
        # (10 + 5) / 5 = 3.0
        self.assertAlmostEqual(row["trips_per_day"], 3.0, places=5)

    # ------------------------------------------------------------------
    # Average metrics
    # ------------------------------------------------------------------

    def test_avg_trip_duration_and_distance(self) -> None:
        """Average duration and distance are simple means of service trips."""
        predictor = self._make_predictor_with_predictions(
            [
                {
                    "trip_id": "T1",
                    "trip_type": "service",
                    "agency_id": "A1",
                    "trip_count": 5,
                    "trip_duration_minutes": 20.0,
                    "miles": 4.0,
                },
                {
                    "trip_id": "T2",
                    "trip_type": "service",
                    "agency_id": "A1",
                    "trip_count": 5,
                    "trip_duration_minutes": 40.0,
                    "miles": 8.0,
                },
            ],
            total_service_days=5,
        )

        overview = predictor.get_agency_overview()

        row = overview[overview["agency_id"] == "A1"].iloc[0]
        self.assertAlmostEqual(row["avg_trip_duration_minutes"], 30.0, places=5)
        self.assertAlmostEqual(row["avg_trip_distance_miles"], 6.0, places=5)

    # ------------------------------------------------------------------
    # Deadhead trips excluded
    # ------------------------------------------------------------------

    def test_deadhead_trips_are_excluded(self) -> None:
        """Only service trips contribute to agency overview metrics."""
        predictor = self._make_predictor_with_predictions(
            [
                {
                    "trip_id": "T1",
                    "trip_type": "service",
                    "agency_id": "A1",
                    "trip_count": 5,
                    "trip_duration_minutes": 30.0,
                    "miles": 10.0,
                },
                {
                    "trip_id": "DH1",
                    "trip_type": "pull-out",
                    "agency_id": "A1",
                    "trip_count": 5,
                    "trip_duration_minutes": 5.0,
                    "miles": 1.0,
                },
            ],
            total_service_days=5,
        )

        overview = predictor.get_agency_overview()

        row = overview[overview["agency_id"] == "A1"].iloc[0]
        # Only T1 (service trip) should count
        self.assertAlmostEqual(row["trips_per_day"], 1.0, places=5)
        self.assertAlmostEqual(row["avg_trip_duration_minutes"], 30.0, places=5)
        self.assertAlmostEqual(row["avg_trip_distance_miles"], 10.0, places=5)

    # ------------------------------------------------------------------
    # Missing agency_id
    # ------------------------------------------------------------------

    def test_missing_agency_id_mapped_to_unknown(self) -> None:
        """NaN agency_id is replaced with 'unknown'."""
        predictor = self._make_predictor_with_predictions(
            [
                {
                    "trip_id": "T1",
                    "trip_type": "service",
                    "agency_id": None,
                    "trip_count": 5,
                    "trip_duration_minutes": 30.0,
                    "miles": 10.0,
                }
            ],
            total_service_days=5,
        )

        overview = predictor.get_agency_overview()

        self.assertEqual(len(overview), 1)
        self.assertEqual(overview.iloc[0]["agency_id"], "unknown")

    # ------------------------------------------------------------------
    # Agency name merge
    # ------------------------------------------------------------------

    def test_agency_name_merged_from_feed(self) -> None:
        """agency_name should be present when feed.agency is available."""
        agency_df = pd.DataFrame(
            {"agency_id": ["A1"], "agency_name": ["Test Agency"]}
        )
        predictor = self._make_predictor_with_predictions(
            [
                {
                    "trip_id": "T1",
                    "trip_type": "service",
                    "agency_id": "A1",
                    "trip_count": 5,
                    "trip_duration_minutes": 30.0,
                    "miles": 10.0,
                }
            ],
            agency_df=agency_df,
            total_service_days=5,
        )

        overview = predictor.get_agency_overview()

        self.assertIn("agency_name", overview.columns)
        self.assertEqual(overview.iloc[0]["agency_name"], "Test Agency")

    # ------------------------------------------------------------------
    # No predictions yet
    # ------------------------------------------------------------------

    def test_raises_when_no_predictions(self) -> None:
        """KeyError is raised when predict_energy() has not been called."""
        predictor = GTFSEnergyPredictor("/tmp/fake_gtfs")
        with self.assertRaises(KeyError):
            predictor.get_agency_overview()


if __name__ == "__main__":
    unittest.main()
