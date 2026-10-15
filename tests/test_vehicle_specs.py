"""Tests for vehicle capacity specs and block feasibility flags."""

import unittest

import pandas as pd

from routee.transit.results import add_block_feasibility, aggregate_block_predictions
from routee.transit.vehicle_specs import (
    VehicleSpec,
    load_vehicle_specs,
    normalize_energy_unit,
    vehicle_specs_table,
)


def _spec(name: str, capacity: float, unit: str, fraction: float) -> VehicleSpec:
    return VehicleSpec(
        name=name,
        powertrain="bev",
        capacity=capacity,
        capacity_unit=unit,
        usable_capacity_fraction=fraction,
        mass_estimate_lbs=32000.0,
    )


class TestVehicleSpecs(unittest.TestCase):
    def test_bundled_models_declare_comparable_capacities(self) -> None:
        specs = load_vehicle_specs()
        self.assertTrue(specs)
        for name, spec in specs.items():
            self.assertEqual(name, spec.name)
            self.assertGreater(spec.capacity, 0)
            self.assertIn(spec.capacity_unit, {"kWh", "gallons_diesel"})
            self.assertGreater(spec.usable_capacity_fraction, 0)
            self.assertLessEqual(spec.usable_capacity_fraction, 1)

    def test_filter_and_usable_capacity(self) -> None:
        specs = load_vehicle_specs(["Transit_Bus_Electric_40ft_300kWh"])
        self.assertEqual(list(specs), ["Transit_Bus_Electric_40ft_300kWh"])
        spec = specs["Transit_Bus_Electric_40ft_300kWh"]
        self.assertEqual(spec.capacity, 300.0)
        self.assertEqual(spec.capacity_unit, "kWh")
        self.assertAlmostEqual(
            spec.usable_capacity, 300.0 * spec.usable_capacity_fraction
        )

    def test_unit_aliases_normalize(self) -> None:
        for alias in ("kilowatt_hours", "kilowatt hour", "kWh"):
            self.assertEqual(normalize_energy_unit(alias), "kWh")
        for alias in ("gallons_diesel", "GDE", "gallons diesel equivalent"):
            self.assertEqual(normalize_energy_unit(alias), "gallons_diesel")
        with self.assertRaises(ValueError):
            normalize_energy_unit("furlongs")

    def test_specs_table_schema(self) -> None:
        table = vehicle_specs_table(load_vehicle_specs())
        self.assertEqual(
            list(table.columns),
            [
                "vehicle",
                "powertrain",
                "mass_estimate_lbs",
                "capacity",
                "capacity_unit",
                "usable_capacity_fraction",
                "usable_capacity",
            ],
        )
        self.assertEqual(table["vehicle"].tolist(), sorted(table["vehicle"]))
        pd.testing.assert_series_equal(
            table["capacity"] * table["usable_capacity_fraction"],
            table["usable_capacity"],
            check_names=False,
        )


class TestBlockFeasibility(unittest.TestCase):
    def setUp(self) -> None:
        self.blocks = pd.DataFrame(
            {
                "vehicle": ["bus", "bus", "unknown"],
                "date": pd.to_datetime(["2026-01-05"] * 3),
                "block_id": ["1", "2", "3"],
                "energy_used_without_deadhead": [100.0, 230.0, 10.0],
                "energy_used_with_deadhead": [120.0, 260.0, 20.0],
                "energy_unit": ["kWh"] * 3,
            }
        )
        self.specs = {"bus": _spec("bus", 300.0, "kWh", 0.8)}

    def test_flags_compare_against_usable_capacity(self) -> None:
        result = add_block_feasibility(self.blocks, self.specs)
        self.assertNotIn("usable_capacity", result.columns)
        self.assertEqual(result["feasible_without_deadhead"].tolist()[:2], [True, True])
        self.assertEqual(result["feasible_with_deadhead"].tolist()[:2], [True, False])

    def test_unknown_vehicle_yields_null_flags(self) -> None:
        result = add_block_feasibility(self.blocks, self.specs)
        self.assertTrue(pd.isna(result["feasible_with_deadhead"].iloc[2]))
        self.assertTrue(pd.isna(result["feasible_without_deadhead"].iloc[2]))
        self.assertEqual(result["feasible_with_deadhead"].dtype, "boolean")

    def test_mismatched_units_raise(self) -> None:
        blocks = self.blocks.assign(energy_unit="gallons_diesel")
        with self.assertRaises(ValueError):
            add_block_feasibility(blocks, self.specs)

    def test_aggregation_adds_feasibility_columns(self) -> None:
        trips = pd.DataFrame(
            {
                "vehicle": ["bus", "bus"],
                "date": pd.to_datetime(["2026-01-05"] * 2),
                "block_id": ["1", "1"],
                "trip_id": ["T1", "T2"],
                "trip_type": ["service", "pull-in"],
                "route_id": ["101", None],
                "agency_id": ["A", "A"],
                "energy_used": [230.0, 20.0],
                "energy_unit": ["kWh", "kWh"],
                "trip_is_within_gtfs_scope": [True, True],
            }
        )
        blocks = aggregate_block_predictions(trips, vehicle_specs=self.specs)
        self.assertNotIn("trip_is_within_gtfs_scope", blocks.columns)
        self.assertEqual(blocks["feasible_without_deadhead"].tolist(), [True])
        self.assertEqual(blocks["feasible_with_deadhead"].tolist(), [False])

    def test_empty_input_keeps_schema(self) -> None:
        empty = aggregate_block_predictions(pd.DataFrame({"trip_id": []}))
        for column in ("feasible_with_deadhead", "feasible_without_deadhead"):
            self.assertIn(column, empty.columns)
        self.assertTrue(empty.empty)


class TestMultiAgencyBlocks(unittest.TestCase):
    def _trips(self, agencies: list[str], dates: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "vehicle": ["bus"] * len(agencies) * len(dates),
                "date": pd.to_datetime([date for date in dates for _ in agencies]),
                "block_id": ["1"] * len(agencies) * len(dates),
                "trip_id": [f"T{i}" for i in range(len(agencies) * len(dates))],
                "trip_type": ["service"] * len(agencies) * len(dates),
                "route_id": ["101"] * len(agencies) * len(dates),
                "agency_id": agencies * len(dates),
                "energy_used": [10.0] * len(agencies) * len(dates),
                "energy_unit": ["kWh"] * len(agencies) * len(dates),
            }
        )

    def test_all_agencies_recorded_sorted_and_deduplicated(self) -> None:
        blocks = aggregate_block_predictions(
            self._trips(["B", "A", "B"], ["2026-01-05"])
        )
        self.assertEqual(blocks["agency_ids"].tolist(), [["A", "B"]])

    def test_single_agency_is_a_one_element_list(self) -> None:
        blocks = aggregate_block_predictions(self._trips(["A", "A"], ["2026-01-05"]))
        self.assertEqual(blocks["agency_ids"].tolist(), [["A"]])

    def test_no_warning_for_multi_agency_blocks(self) -> None:
        trips = self._trips(["A", "B"], ["2026-01-05", "2026-01-06", "2026-01-07"])
        with self.assertNoLogs("routee.transit.results", level="WARNING"):
            blocks = aggregate_block_predictions(trips)
        self.assertEqual(len(blocks), 3)
        self.assertTrue(all(a == ["A", "B"] for a in blocks["agency_ids"]))


class TestDeadheadOnlyBlocks(unittest.TestCase):
    def _trips(self, trip_types: list[str], dates: list[str]) -> pd.DataFrame:
        n = len(trip_types) * len(dates)
        return pd.DataFrame(
            {
                "vehicle": ["bus"] * n,
                "date": pd.to_datetime([date for date in dates for _ in trip_types]),
                "block_id": ["1"] * n,
                "trip_id": [f"T{i}" for i in range(n)],
                "trip_type": trip_types * len(dates),
                "route_id": ["101"] * n,
                "agency_id": ["A"] * n,
                "energy_used": [10.0] * n,
                "energy_unit": ["kWh"] * n,
            }
        )

    def test_block_without_revenue_service_is_dropped(self) -> None:
        trips = self._trips(["pull-out", "pull-in"], ["2026-01-05"])
        with self.assertLogs("routee.transit.results", level="WARNING") as logs:
            blocks = aggregate_block_predictions(trips)
        self.assertTrue(blocks.empty)
        self.assertIn("Dropped 1 block(s)", logs.output[0])

    def test_drop_warning_emitted_once_across_dates(self) -> None:
        trips = self._trips(["pull-in"], ["2026-01-05", "2026-01-06", "2026-01-07"])
        with self.assertLogs("routee.transit.results", level="WARNING") as logs:
            aggregate_block_predictions(trips)
        self.assertEqual(len(logs.output), 1)

    def test_block_keeps_deadhead_when_service_survives(self) -> None:
        trips = self._trips(["service", "pull-in"], ["2026-01-05"])
        with self.assertNoLogs("routee.transit.results", level="WARNING"):
            blocks = aggregate_block_predictions(trips)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks["energy_used_without_deadhead"].tolist(), [10.0])
        self.assertEqual(blocks["energy_used_with_deadhead"].tolist(), [20.0])


if __name__ == "__main__":
    unittest.main()
