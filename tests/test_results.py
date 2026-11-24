"""Tests for per-route result summaries exported as route shapes."""

import unittest

import pandas as pd

from routee.transit.results import (
    aggregate_block_predictions,
    block_result_columns,
    build_route_shapes,
)

SHAPES = pd.DataFrame(
    {
        "shape_id": ["SH1", "SH1", "SH2", "SH2", "SH3", "SH3", "SHD", "SHD"],
        "shape_pt_lat": [39.7, 39.71, 39.8, 39.81, 39.9, 39.91, 40.0, 40.01],
        "shape_pt_lon": [
            -105.0,
            -104.99,
            -105.1,
            -105.09,
            -105.2,
            -105.19,
            -105.3,
            -105.29,
        ],
        "shape_pt_sequence": [1, 2, 1, 2, 1, 2, 1, 2],
    }
)

TRIPS = pd.DataFrame(
    {
        "trip_id": ["T1", "T2", "T3", "T4", "D1"],
        "route_id": ["101", "101", "101", "102", "deadhead_A_to_B"],
        "shape_id": ["SH1", "SH1", "SH2", "SH3", "SHD"],
        "trip_type": ["service"] * 4 + ["mid_block_deadhead"],
        "route_short_name": ["101", "101", "101", "102", "deadhead_A_to_B"],
        "agency_id": ["A", "A", "A", "A", "A"],
        "route_color": ["FF0000", "FF0000", "FF0000", "00FF00", None],
    }
)

# T1 operates on two dates, so route 101 has three electric trip days.
PREDICTIONS = pd.DataFrame(
    {
        "trip_id": ["T1", "T1", "T2", "T3", "T4", "T4", "D1"],
        "route_id": ["101", "101", "101", "101", "102", "102", "deadhead_A_to_B"],
        "vehicle": ["EV", "EV", "EV", "EV", "EV", "Diesel", "EV"],
        "energy_used": [10.0, 14.0, 20.0, 4.0, 8.0, 2.0, 99.0],
        "miles": [5.0, 5.0, 5.0, 2.0, 4.0, 4.0, 1.0],
        "energy_unit": ["kWh", "kWh", "kWh", "kWh", "kWh", "gallons", "kWh"],
        "trip_type": ["service"] * 6 + ["mid_block_deadhead"],
    }
)


class TestBuildRouteShapes(unittest.TestCase):
    def test_most_common_shape_represents_each_route(self) -> None:
        routes = build_route_shapes(TRIPS, SHAPES, PREDICTIONS).set_index("route_id")
        self.assertEqual(sorted(routes.index), ["101", "102"])
        self.assertEqual(routes.loc["101", "shape_id"], "SH1")
        self.assertEqual(routes.loc["102", "shape_id"], "SH3")
        self.assertEqual(routes.crs, "EPSG:4326")
        self.assertEqual(
            list(routes.loc["101", "geometry"].coords),
            [(-105.0, 39.7), (-104.99, 39.71)],
        )
        self.assertEqual(routes.loc["101", "route_color"], "FF0000")
        self.assertEqual(routes.loc["102", "agency_id"], "A")

    def test_energy_averages_over_trip_days_by_vehicle(self) -> None:
        routes = build_route_shapes(TRIPS, SHAPES, PREDICTIONS).set_index("route_id")
        self.assertEqual(
            routes.loc["101", "mean_energy_used"],
            {"EV": {"value": 48.0 / 4, "unit": "kWh"}},
        )
        self.assertEqual(
            routes.loc["101", "mean_efficiency"],
            {"EV": {"value": 48.0 / 17.0, "unit": "kWh/mi"}},
        )
        self.assertEqual(
            routes.loc["102", "mean_efficiency"],
            {
                "EV": {"value": 8.0 / 4.0, "unit": "kWh/mi"},
                "Diesel": {"value": 4.0 / 2.0, "unit": "mpg_diesel"},
            },
        )

    def test_optional_route_metadata_is_skipped_when_absent(self) -> None:
        trips = TRIPS.drop(columns=["route_color", "agency_id"])
        routes = build_route_shapes(trips, SHAPES, PREDICTIONS)
        self.assertEqual(
            list(routes.columns),
            [
                "route_id",
                "shape_id",
                "route_short_name",
                "mean_energy_used",
                "mean_efficiency",
                "geometry",
            ],
        )

    def test_empty_inputs_keep_schema(self) -> None:
        routes = build_route_shapes(TRIPS.iloc[:0], SHAPES, PREDICTIONS)
        self.assertTrue(routes.empty)
        self.assertEqual(routes.crs, "EPSG:4326")
        self.assertIn("mean_efficiency", routes.columns)
        deadhead_only = PREDICTIONS.loc[PREDICTIONS["trip_type"].ne("service")]
        self.assertTrue(build_route_shapes(TRIPS, SHAPES, deadhead_only).empty)


class TestExplodedBlockRoutes(unittest.TestCase):
    def _trips(
        self, route_ids: list[str | None], agency_ids: list[str | None]
    ) -> pd.DataFrame:
        count = len(route_ids)
        return pd.DataFrame(
            {
                "vehicle": ["bus"] * count,
                "date": pd.to_datetime(["2026-01-05"] * count),
                "block_id": ["1"] * count,
                "trip_id": [f"T{index}" for index in range(count)],
                "trip_type": ["service"] * count,
                "route_id": route_ids,
                "agency_id": agency_ids,
                "energy_used": [10.0] * count,
                "energy_unit": ["kWh"] * count,
            }
        )

    def test_one_row_per_route_repeats_block_totals(self) -> None:
        trips = self._trips(["102", "101", "101"], ["B", "A", "A"])
        blocks = aggregate_block_predictions(trips, explode_routes=True)
        self.assertEqual(blocks["route_id"].tolist(), ["101", "102"])
        self.assertEqual(blocks["agency_id"].tolist(), ["A", "B"])
        self.assertEqual(blocks["energy_used_with_deadhead"].tolist(), [30.0, 30.0])
        self.assertEqual(blocks["block_id"].tolist(), ["1", "1"])
        for column in ("route_ids", "agency_ids"):
            self.assertNotIn(column, blocks.columns)

    def test_default_keeps_rolled_up_lists(self) -> None:
        blocks = aggregate_block_predictions(self._trips(["102", "101"], ["B", "A"]))
        self.assertEqual(blocks["route_ids"].tolist(), [["101", "102"]])
        self.assertEqual(len(blocks), 1)

    def test_deadhead_rows_do_not_add_routes(self) -> None:
        trips = self._trips(["101", None], ["A", None])
        trips.loc[1, "trip_type"] = "pull-in"
        blocks = aggregate_block_predictions(trips, explode_routes=True)
        self.assertEqual(blocks["route_id"].tolist(), ["101"])
        self.assertEqual(blocks["energy_used_without_deadhead"].tolist(), [10.0])
        self.assertEqual(blocks["energy_used_with_deadhead"].tolist(), [20.0])

    def test_missing_route_metadata_yields_null_identifiers(self) -> None:
        trips = self._trips(["101"], ["A"]).drop(columns=["route_id", "agency_id"])
        blocks = aggregate_block_predictions(trips, explode_routes=True)
        self.assertEqual(len(blocks), 1)
        self.assertTrue(blocks["route_id"].isna().all())
        self.assertTrue(blocks["agency_id"].isna().all())

    def test_empty_input_keeps_exploded_schema(self) -> None:
        empty = aggregate_block_predictions(
            pd.DataFrame({"trip_id": []}), explode_routes=True
        )
        self.assertTrue(empty.empty)
        self.assertEqual(
            list(empty.columns),
            block_result_columns(explode_routes=True)
            + ["feasible_without_deadhead", "feasible_with_deadhead"],
        )


if __name__ == "__main__":
    unittest.main()
