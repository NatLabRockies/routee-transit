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


if __name__ == "__main__":
    unittest.main()
