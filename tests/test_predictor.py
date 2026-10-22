import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

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

    def test_filter_trips_by_date(self) -> None:
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
        self.assertEqual(self.predictor._service_date, pd.Timestamp("2023-01-01"))

    def test_missing_ids_preserve_route_filtered_trips(self) -> None:
        self.predictor.feed = MagicMock()
        self.predictor.trips = pd.DataFrame(
            {
                "trip_id": ["T1", "T2", "T3", "T4"],
                "block_id": ["B1", "B1", None, " "],
                "route_short_name": ["R1", "R2", "R1", "R2"],
                "service_id": ["S1"] * 4,
                "shape_id": ["SH1"] * 4,
            }
        )
        self.predictor.feed.shapes = pd.DataFrame({"shape_id": ["SH1"]})
        self.predictor.filter_trips(routes=["R1"], use_block_filter=True)
        self.assertEqual(self.predictor.trips["trip_id"].tolist(), ["T3"])
        self.assertIsNone(self.predictor._prepare_mid_block_deadhead())
        self.assertIsNone(self.predictor._prepare_depot_deadhead())

    def test_absent_ids_skip_deadhead_without_dropping_trips(self) -> None:
        self.predictor.feed = MagicMock()
        self.predictor.trips = pd.DataFrame(
            {
                "trip_id": ["T1"],
                "service_id": ["S1"],
                "route_short_name": ["R1"],
                "shape_id": ["SH1"],
            }
        )
        self.predictor.feed.shapes = pd.DataFrame({"shape_id": ["SH1"]})
        self.predictor.filter_trips(routes=["R1"], use_block_filter=True)
        self.assertIsNone(self.predictor._prepare_mid_block_deadhead())
        self.assertIsNone(self.predictor._prepare_depot_deadhead())
        self.assertEqual(len(self.predictor.trips), 1)

    def test_block_getter_before_prediction(self) -> None:
        with self.assertRaisesRegex(KeyError, "Call predict_energy"):
            self.predictor.get_block_predictions()

    def test_filter_trips_trip_level_default(self) -> None:
        """Trip-level filtering keeps individual trips even if the block has other routes."""
        self.predictor.feed = MagicMock()
        self.predictor.trips = pd.DataFrame(
            {
                "trip_id": ["T1", "T2", "T3"],
                "block_id": ["B1", "B1", "B2"],
                "route_short_name": ["R1", "R2", "R1"],
                "service_id": ["S1", "S1", "S1"],
                "shape_id": ["SH1", "SH2", "SH3"],
            }
        )
        self.predictor.feed.shapes = pd.DataFrame({"shape_id": ["SH1", "SH2", "SH3"]})

        self.predictor.filter_trips(routes=["R1"])

        # Should keep T1 and T3 (both on R1) even though T1's block also has R2
        self.assertEqual(len(self.predictor.trips), 2)
        self.assertListEqual(
            sorted(self.predictor.trips["trip_id"].tolist()), ["T1", "T3"]
        )

    def test_filter_trips_block_level(self) -> None:
        """Block-level filtering excludes blocks that contain trips from other routes."""
        self.predictor.feed = MagicMock()
        self.predictor.trips = pd.DataFrame(
            {
                "trip_id": ["T1", "T2", "T3"],
                "block_id": ["B1", "B1", "B2"],
                "route_short_name": ["R1", "R2", "R1"],
                "service_id": ["S1", "S1", "S1"],
                "shape_id": ["SH1", "SH2", "SH3"],
            }
        )
        self.predictor.feed.shapes = pd.DataFrame({"shape_id": ["SH1", "SH2", "SH3"]})

        self.predictor.filter_trips(routes=["R1"], use_block_filter=True)

        # Block B1 has R2 so it is excluded entirely; only B2/T3 remains
        self.assertEqual(len(self.predictor.trips), 1)
        self.assertEqual(self.predictor.trips.iloc[0]["trip_id"], "T3")

    def test_filter_trips_block_level_error_message(self) -> None:
        """Block-level filtering gives a helpful error when all trips are excluded."""
        self.predictor.feed = MagicMock()
        self.predictor.trips = pd.DataFrame(
            {
                "trip_id": ["T1", "T2"],
                "block_id": ["B1", "B1"],
                "route_short_name": ["R1", "R2"],
                "service_id": ["S1", "S1"],
                "shape_id": ["SH1", "SH2"],
            }
        )
        self.predictor.feed.shapes = pd.DataFrame({"shape_id": ["SH1", "SH2"]})

        with self.assertRaises(ValueError) as ctx:
            self.predictor.filter_trips(routes=["R1"], use_block_filter=True)

        self.assertIn("block-level", str(ctx.exception))
        self.assertIn("interlined", str(ctx.exception))
        # Should mention the count of trip-level matches
        self.assertIn("1 trip(s) match at the trip level", str(ctx.exception))

    def test_filter_trips_block_level_no_trips_at_all(self) -> None:
        """Block-level filtering with no matching trips gives a generic error."""
        self.predictor.feed = MagicMock()
        self.predictor.trips = pd.DataFrame(
            {
                "trip_id": ["T1", "T2"],
                "block_id": ["B1", "B1"],
                "route_short_name": ["R1", "R2"],
                "service_id": ["S1", "S1"],
                "shape_id": ["SH1", "SH2"],
            }
        )
        self.predictor.feed.shapes = pd.DataFrame({"shape_id": ["SH1", "SH2"]})

        with self.assertRaises(ValueError) as ctx:
            self.predictor.filter_trips(routes=["R99"], use_block_filter=True)

        # No trip-level matches either, so the generic error should be used
        self.assertIn("No trips found", str(ctx.exception))

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
        self.predictor.feed.get_service_ids_all_dates.return_value = pd.DataFrame(
            {"service_id": ["S1"], "date": pd.to_datetime(["2026-01-05"])}
        )
        self.predictor.trips = pd.DataFrame(
            {
                "trip_id": ["T1", "T2"],
                "shape_id": ["SH1", "SH1"],
                "service_id": ["S1", "S1"],
                "route_short_name": ["101", "101"],
                "route_desc": ["Main Line", "Main Line"],
                "route_type": [3, 3],
                "block_id": ["001", "001"],
                "trip_type": ["service", "service"],
                "route_id": ["101", "101"],
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

    def _run_predict_energy(
        self,
        vehicle_models: list[str],
        add_hvac: bool = False,
        scale_to_year: bool = False,
    ) -> dict[str, pd.DataFrame]:
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
            return self.predictor.predict_energy(
                add_hvac=add_hvac, scale_to_year=scale_to_year
            )

    def test_daily_models_and_block_reconciliation(self) -> None:
        assert self.predictor.feed is not None
        self.predictor.feed.get_service_ids_all_dates.return_value = pd.DataFrame(
            {
                "service_id": ["S1", "S1"],
                "date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            }
        )
        electric = "Transit_Bus_Battery_Electric"
        diesel = "Transit_Bus_Diesel"
        self.mock_app.run_calculate_path.side_effect = [
            self._make_compass_result(10, 5, "trip_energy_electric"),
            self._make_compass_result(20, 5, "trip_energy_liquid"),
        ]
        results = self._run_predict_energy([electric, diesel])
        self.assertEqual(len(results["trip"]), 8)
        self.assertEqual(len(results["block"]), 4)
        expected = (
            results["trip"]
            .groupby(["vehicle", "date", "block_id"])["energy_used"]
            .sum()
        )
        actual = results["block"].set_index(["vehicle", "date", "block_id"])[
            "energy_used_with_deadhead"
        ]
        pd.testing.assert_series_equal(expected, actual.sort_index(), check_names=False)
        self.assertEqual(len(self.predictor.get_block_predictions(diesel)), 2)
        self.assertEqual(
            results["trip"].duplicated(["vehicle", "trip_id", "date"]).sum(), 0
        )
        self.mock_app.run_calculate_path.side_effect = None
        self.mock_app.run_calculate_path.return_value = self._make_compass_result(
            10, 5, "trip_energy_electric"
        )
        results = self._run_predict_energy([electric])
        self.assertNotIn(f"{diesel}_block", results)
        with self.assertRaises(KeyError):
            self.predictor.get_block_predictions(diesel)

    def test_daily_hvac_join_and_deadhead(self) -> None:
        assert self.predictor.feed is not None
        self.predictor.feed.get_service_ids_all_dates.return_value = pd.DataFrame(
            {
                "service_id": ["S1", "S1"],
                "date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            }
        )
        self.predictor.trips.loc[1, "trip_type"] = "pull-in"
        self.mock_app.run_calculate_path.return_value = self._make_compass_result(
            10, 5, "trip_energy_electric"
        )
        hvac = pd.DataFrame(
            {
                "trip_id": ["T1", "T1", "T2"],
                "date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-05"]),
                "hvac_energy_kWh": [1.0, 2.0, 3.0],
                "trip_is_within_gtfs_scope": [True] * 3,
                "scenario": ["TMY"] * 3,
            }
        )
        with patch("routee.transit.predictor.add_HVAC_energy", return_value=hvac):
            results = self._run_predict_energy(
                ["Transit_Bus_Battery_Electric"], add_hvac=True
            )
        self.assertEqual(len(results["trip"]), 4)
        block = results["block"].sort_values("date")
        self.assertEqual(block["energy_used_without_deadhead"].tolist(), [11, 12])
        self.assertEqual(block["energy_used_with_deadhead"].tolist(), [24, 22])
        self.assertTrue(results["trip"]["trip_is_within_gtfs_scope"].all())

    def test_hvac_computed_once_for_multiple_electric_models(self) -> None:
        assert self.predictor.feed is not None
        self.predictor.feed.get_service_ids_all_dates.return_value = pd.DataFrame(
            {"service_id": ["S1"], "date": pd.to_datetime(["2026-01-05"])}
        )
        self.mock_app.run_calculate_path.return_value = self._make_compass_result(
            10, 5, "trip_energy_electric"
        )
        hvac = pd.DataFrame(
            {
                "trip_id": ["T1", "T2"],
                "date": pd.to_datetime(["2026-01-05"] * 2),
                "hvac_energy_kWh": [1.0, 3.0],
                "trip_is_within_gtfs_scope": [True] * 2,
                "scenario": ["TMY"] * 2,
            }
        )
        with patch(
            "routee.transit.predictor.add_HVAC_energy", return_value=hvac
        ) as mock_hvac:
            results = self._run_predict_energy(
                [
                    "Transit_Bus_Electric_40ft_300kWh",
                    "Transit_Bus_Electric_60ft_600kWh",
                    "Transit_Bus_Diesel_40ft",
                ],
                add_hvac=True,
            )
        self.assertEqual(mock_hvac.call_count, 1)
        for model in (
            "Transit_Bus_Electric_40ft_300kWh",
            "Transit_Bus_Electric_60ft_600kWh",
        ):
            trips = results[f"{model}_trip"].sort_values("trip_id")
            self.assertEqual(trips["hvac_energy_kWh"].tolist(), [1.0, 3.0])
            self.assertEqual(trips["energy_used"].tolist(), [11.0, 13.0])
        diesel = results["Transit_Bus_Diesel_40ft_trip"]
        self.assertNotIn("hvac_energy_kWh", diesel.columns)

    def test_hvac_skipped_when_no_electric_model_requested(self) -> None:
        self.mock_app.run_calculate_path.return_value = self._make_compass_result(
            10, 5, "trip_energy_liquid"
        )
        with patch("routee.transit.predictor.add_HVAC_energy") as mock_hvac:
            self._run_predict_energy(["Transit_Bus_Diesel_40ft"], add_hvac=True)
        mock_hvac.assert_not_called()

    def test_annual_projection_and_single_date_without_hvac(self) -> None:
        assert self.predictor.feed is not None
        self.predictor.feed.get_typical_service_by_weekday.return_value = pd.DataFrame(
            {"service_id": [["S1"]]}, index=["monday"]
        )
        self.mock_app.run_calculate_path.return_value = self._make_compass_result(
            10, 5, "trip_energy_liquid"
        )
        results = self._run_predict_energy(["Transit_Bus_Diesel"], scale_to_year=True)
        self.assertEqual(len(results["block"]), 52)
        self.assertNotIn("trip_is_within_gtfs_scope", results["block"].columns)
        self.assertEqual(results["trip"]["trip_is_within_gtfs_scope"].sum(), 2)
        self.predictor._service_date = pd.Timestamp("2026-01-05")
        results = self._run_predict_energy(["Transit_Bus_Diesel"], scale_to_year=True)
        self.assertEqual(len(results["trip"]), 2)
        self.assertEqual(len(results["block"]), 1)

    def test_exports_repeat_safely_and_preserve_results(self) -> None:
        self.predictor.dataset_id = "dataset"
        self.predictor.feed_id = "feed"
        self.mock_app.run_calculate_path.return_value = self._make_compass_result(
            10, 5, "trip_energy_electric"
        )
        results = self._run_predict_energy(["Transit_Bus_Battery_Electric"])
        original_trip = results["trip"].copy(deep=True)
        original_block = results["block"].copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            self.predictor.save_results(directory, save_tods=False)
            self.predictor.save_results(directory, save_tods=False)
            blocks = pd.read_csv(
                Path(directory) / "block_energy_predictions.csv",
                dtype={"block_id": str},
            )
            self.assertEqual(blocks["block_id"].tolist(), ["001"])
            self.assertEqual(json.loads(blocks.iloc[0]["route_ids"]), ["101"])
            self.assertEqual(blocks.iloc[0]["date"], "2026-01-05")
            for column in (
                "feed_id",
                "dataset_id",
                "usable_capacity",
                "trip_is_within_gtfs_scope",
            ):
                self.assertNotIn(column, blocks.columns)
            trips = pd.read_csv(Path(directory) / "trip_energy_predictions.csv")
            self.assertEqual(trips.iloc[0]["feed_id"], "feed")
            self.assertEqual(trips.iloc[0]["dataset_id"], "dataset")
            self.assertEqual(
                blocks.iloc[0]["energy_used_with_deadhead"], trips["energy_used"].sum()
            )
            routes = json.loads((Path(directory) / "routes.geojson").read_text())
            self.assertEqual(len(routes["features"]), 1)
            properties = routes["features"][0]["properties"]
            self.assertEqual(properties["route_id"], "101")
            self.assertEqual(properties["shape_id"], "SH1")
            self.assertEqual(
                properties["mean_energy_used"]["Transit_Bus_Battery_Electric"],
                {"value": trips["energy_used"].mean(), "unit": "kWh"},
            )
            self.assertEqual(
                properties["mean_efficiency"]["Transit_Bus_Battery_Electric"]["unit"],
                "kWh/mi",
            )
        pd.testing.assert_frame_equal(original_trip, results["trip"])
        pd.testing.assert_frame_equal(original_block, results["block"])
        with tempfile.TemporaryDirectory() as directory:
            self.predictor.save_results(
                directory,
                save_tods=False,
                save_block_level=False,
                save_route_shapes=False,
            )
            self.assertFalse(
                (Path(directory) / "block_energy_predictions.csv").exists()
            )
            self.assertFalse((Path(directory) / "routes.geojson").exists())

    def test_exploded_block_routes_export_scalar_identifiers(self) -> None:
        self.predictor.explode_block_routes = True
        self.predictor.trips["route_id"] = ["101", "102"]
        self.mock_app.run_calculate_path.return_value = self._make_compass_result(
            10, 5, "trip_energy_electric"
        )
        results = self._run_predict_energy(["Transit_Bus_Battery_Electric"])
        self.assertEqual(results["block"]["route_id"].tolist(), ["101", "102"])
        with tempfile.TemporaryDirectory() as directory:
            self.predictor.save_results(directory, save_tods=False)
            blocks = pd.read_csv(
                Path(directory) / "block_energy_predictions.csv",
                dtype={"block_id": str, "route_id": str},
            )
        self.assertEqual(blocks["route_id"].tolist(), ["101", "102"])
        self.assertNotIn("route_ids", blocks.columns)
        self.assertEqual(blocks["block_id"].tolist(), ["001", "001"])

    def test_empty_service_calendar(self) -> None:
        assert self.predictor.feed is not None
        self.predictor.feed.get_service_ids_all_dates.return_value = pd.DataFrame(
            columns=["date", "service_id"]
        )
        self.mock_app.run_calculate_path.return_value = self._make_compass_result(
            10, 5, "trip_energy_electric"
        )
        results = self._run_predict_energy(["Transit_Bus_Battery_Electric"])
        self.assertTrue(results["trip"].empty)
        self.assertTrue(results["block"].empty)

    def test_all_missing_blocks_still_export_trips(self) -> None:
        self.predictor.trips = self.predictor.trips.drop(columns="block_id")
        self.mock_app.run_calculate_path.return_value = self._make_compass_result(
            10, 5, "trip_energy_electric"
        )
        with self.assertLogs("routee.transit.predictor", level="WARNING") as captured:
            results = self._run_predict_energy(["Transit_Bus_Battery_Electric"])
        self.assertIn("Excluded 2 trips", " ".join(captured.output))
        self.assertEqual(len(results["trip"]), 2)
        self.assertTrue(self.predictor.get_block_predictions().empty)
        self.assertTrue(
            self.predictor.get_block_predictions("Transit_Bus_Battery_Electric").empty
        )
        with tempfile.TemporaryDirectory() as directory:
            self.predictor.save_results(directory, save_tods=False)
            exported = pd.read_csv(Path(directory) / "block_energy_predictions.csv")
            self.assertTrue(exported.empty)
            self.assertEqual(
                exported.columns.tolist(), results["block"].columns.tolist()
            )

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

    def test_queries_are_per_shape_not_trip_or_service_date(self) -> None:
        self.predictor.trips["start_time"] = ["08:00:00", "09:00:00"]
        self.mock_app.run_calculate_path.return_value = self._make_compass_result(
            energy_value=1.0,
            distance_value=10.0,
            energy_field="trip_energy_electric",
        )
        self._run_predict_energy(["Transit_Bus_Battery_Electric"])
        self.mock_app.run_calculate_path.assert_called_once()
        queries = self.mock_app.run_calculate_path.call_args.args[0]
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0]["start_time"], "08:00:00")
        self.assertEqual(queries[0]["start_weekday"], "monday")


if __name__ == "__main__":
    unittest.main()


class TestShapeLinkResults(unittest.TestCase):
    """Tests for _get_shape_link_results(): sorting and routed-edge reuse."""

    def setUp(self) -> None:
        self.predictor = GTFSEnergyPredictor("/tmp/fake_gtfs", n_processes=1)
        # Rows deliberately out of shape_pt_sequence order
        self.predictor.shapes = pd.DataFrame(
            {
                "shape_id": ["SH1", "SH1", "SH1", "DH1", "DH1"],
                "shape_pt_lat": [39.72, 39.70, 39.71, 39.70, 39.75],
                "shape_pt_lon": [-104.97, -104.99, -104.98, -104.99, -104.90],
                "shape_pt_sequence": [3, 1, 2, 1, 2],
            }
        )

    def _run(self) -> tuple[pd.DataFrame, list[pd.DataFrame], MagicMock]:
        captured: list[pd.DataFrame] = []

        def fake_pool_map(func: object, groups: list[pd.DataFrame]) -> list[str]:
            captured.extend(groups)
            return [str(g["shape_id"].iloc[0]) for g in groups]

        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.map.side_effect = fake_pool_map
        matched = pd.DataFrame({"shape_id": ["SH1"], "edge_id": [1]})
        with (
            patch("routee.transit.predictor.mp.Pool", return_value=mock_pool),
            patch.object(
                self.predictor, "_match_shapes_to_network", return_value=matched
            ) as mock_match,
        ):
            result = self.predictor._get_shape_link_results()
        return result, captured, mock_match

    def test_shapes_sorted_before_upsampling(self) -> None:
        _, captured, _ = self._run()
        sh1 = next(g for g in captured if g["shape_id"].iloc[0] == "SH1")
        self.assertEqual(sh1["shape_pt_sequence"].tolist(), [1, 2, 3])

    def test_routed_deadhead_shapes_skip_map_matching(self) -> None:
        self.predictor._add_deadhead_route_edges(
            pd.DataFrame(
                {"shape_id": ["DH1", "DH1"], "edge_id": [7, 8], "match_id": [0, 0]}
            )
        )
        result, captured, mock_match = self._run()

        self.assertEqual([g["shape_id"].iloc[0] for g in captured], ["SH1"])
        self.assertEqual(mock_match.call_args[0][0], ["SH1"])
        self.assertEqual(
            result[result["shape_id"] == "DH1"]["edge_id"].tolist(), [7, 8]
        )
        self.assertEqual(result[result["shape_id"] == "SH1"]["edge_id"].tolist(), [1])
        self.assertNotIn("match_id", result.columns)

    def test_route_edges_prefixed_and_replaced(self) -> None:
        edges = pd.DataFrame({"shape_id": ["A"], "edge_id": [1]})
        self.predictor._add_deadhead_route_edges(edges, prefix="to_depot_")
        self.predictor._add_deadhead_route_edges(
            pd.DataFrame({"shape_id": ["A"], "edge_id": [2]}), prefix="to_depot_"
        )
        stored = self.predictor._deadhead_route_edges
        self.assertEqual(stored["shape_id"].tolist(), ["to_depot_A"])
        self.assertEqual(stored["edge_id"].tolist(), [2])
