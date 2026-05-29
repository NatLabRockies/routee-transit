import tempfile
import unittest
from pathlib import Path

import pandas as pd

from routee.transit.gtfs_processing import timedelta_to_gtfs_time
from routee.transit.tods_export import write_tods_deadhead


class TestTimedeltaToHhmmss(unittest.TestCase):
    def test_simple_time(self) -> None:
        td = pd.Timedelta(hours=8, minutes=30, seconds=0)
        self.assertEqual(timedelta_to_gtfs_time(td), "08:30:00")

    def test_past_midnight(self) -> None:
        # GTFS allows times past 24:00 for next-day service
        td = pd.Timedelta(hours=25, minutes=5, seconds=10)
        self.assertEqual(timedelta_to_gtfs_time(td), "25:05:10")

    def test_nan(self) -> None:
        self.assertEqual(timedelta_to_gtfs_time(pd.NaT), "")


class TestWriteTodsDeadhead(unittest.TestCase):
    def setUp(self) -> None:
        self.deadhead_trips = pd.DataFrame(
            {
                "trip_id": [
                    "depot_to_T1",
                    "T2_to_depot",
                    "T3_to_T4",
                ],
                "route_id": [
                    "from_depot_BLK1",
                    "to_depot_BLK1",
                    "deadhead_stop_A_to_stop_B",
                ],
                "service_id": ["svc1", "svc1", "svc2"],
                "block_id": ["BLK1", "BLK1", "BLK2"],
                "shape_id": ["shape_pullout", "shape_pullin", "shape_mid"],
                "trip_type": ["pull-out", "pull-in", "mid_block_deadhead"],
            }
        )

        self.deadhead_stop_times = pd.DataFrame(
            {
                "trip_id": [
                    "depot_to_T1",
                    "depot_to_T1",
                    "T2_to_depot",
                    "T2_to_depot",
                    "T3_to_T4",
                    "T3_to_T4",
                ],
                "stop_sequence": [1, 2, 1, 2, 1, 2],
                "arrival_time": [
                    pd.Timedelta(hours=7),
                    pd.Timedelta(hours=7, minutes=30),
                    pd.Timedelta(hours=17),
                    pd.Timedelta(hours=17, minutes=20),
                    pd.Timedelta(hours=10),
                    pd.Timedelta(hours=10, minutes=15),
                ],
                "departure_time": [
                    pd.Timedelta(hours=7),
                    pd.Timedelta(hours=7, minutes=30),
                    pd.Timedelta(hours=17),
                    pd.Timedelta(hours=17, minutes=20),
                    pd.Timedelta(hours=10),
                    pd.Timedelta(hours=10, minutes=15),
                ],
                "stop_id": [
                    "depot_42",
                    "stop_gtfs_1",
                    "stop_gtfs_2",
                    "depot_42",
                    "stop_gtfs_3",
                    "stop_gtfs_4",
                ],
            }
        )

        # Depot stops only (revenue stops are in gtfs_stops below)
        self.deadhead_stops = pd.DataFrame(
            {
                "stop_id": ["depot_42"],
                "stop_lat": [40.76],
                "stop_lon": [-111.89],
            }
        )

        # Simulate base GTFS stops (revenue stops)
        self.gtfs_stops = pd.DataFrame(
            {
                "stop_id": ["stop_gtfs_1", "stop_gtfs_2", "stop_gtfs_3", "stop_gtfs_4"],
                "stop_lat": [40.75, 40.74, 40.73, 40.72],
                "stop_lon": [-111.88, -111.87, -111.86, -111.85],
            }
        )

        # Minimal shapes for all three deadhead trips
        shape_rows = []
        for shape_id in ["shape_pullout", "shape_pullin", "shape_mid"]:
            for seq, (lat, lon) in enumerate(
                [(40.76, -111.89), (40.755, -111.885), (40.75, -111.88)], start=1
            ):
                shape_rows.append(
                    {
                        "shape_id": shape_id,
                        "shape_pt_lat": lat,
                        "shape_pt_lon": lon,
                        "shape_pt_sequence": seq,
                    }
                )
        self.shapes = pd.DataFrame(shape_rows)

    def _write_and_read(self, output_dir: Path) -> dict[str, pd.DataFrame]:
        write_tods_deadhead(
            deadhead_trips=self.deadhead_trips,
            deadhead_stop_times=self.deadhead_stop_times,
            deadhead_stops=self.deadhead_stops,
            shapes=self.shapes,
            gtfs_stops=self.gtfs_stops,
            output_dir=output_dir,
        )
        files = [
            "trips_supplement.txt",
            "routes_supplement.txt",
            "stops_supplement.txt",
            "stop_times_supplement.txt",
            "shapes_supplement.txt",
        ]
        return {f: pd.read_csv(output_dir / f, dtype=str) for f in files}

    def test_all_files_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dfs = self._write_and_read(Path(tmpdir))
            for fname in dfs:
                self.assertIn(fname, dfs)
                self.assertFalse(dfs[fname].empty, f"{fname} should not be empty")

    def test_trip_type_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dfs = self._write_and_read(Path(tmpdir))
            trips = dfs["trips_supplement.txt"]
            mapping = dict(zip(trips["trip_id"], trips["TODS_trip_type"]))
            self.assertEqual(mapping["depot_to_T1"], "pull-out")
            self.assertEqual(mapping["T2_to_depot"], "pull-back")  # NOT pull-in
            self.assertEqual(mapping["T3_to_T4"], "deadhead")

    def test_depot_stops_have_garage_location_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dfs = self._write_and_read(Path(tmpdir))
            stops = dfs["stops_supplement.txt"]
            depot_rows = stops[stops["stop_id"] == "depot_42"]
            self.assertEqual(len(depot_rows), 1)
            self.assertEqual(depot_rows.iloc[0]["TODS_location_type"], "garage")

    def test_revenue_stops_not_in_stops_supplement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dfs = self._write_and_read(Path(tmpdir))
            stops = dfs["stops_supplement.txt"]
            # None of the base GTFS stop IDs should appear
            for stop_id in self.gtfs_stops["stop_id"]:
                self.assertNotIn(stop_id, stops["stop_id"].values)

    def test_stop_times_arrival_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dfs = self._write_and_read(Path(tmpdir))
            st = dfs["stop_times_supplement.txt"]
            # All arrival_time values should match HH:MM:SS
            import re

            pattern = re.compile(r"^\d{2}:\d{2}:\d{2}$")
            for val in st["arrival_time"]:
                self.assertRegex(val, pattern)

    def test_shapes_supplement_contains_deadhead_shape_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dfs = self._write_and_read(Path(tmpdir))
            shapes = dfs["shapes_supplement.txt"]
            written_ids = set(shapes["shape_id"].unique())
            expected_ids = {"shape_pullout", "shape_pullin", "shape_mid"}
            self.assertEqual(written_ids, expected_ids)

    def test_trip_ids_in_trips_supplement_match_stop_times(self) -> None:
        """Every trip in trips_supplement should have entries in stop_times_supplement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dfs = self._write_and_read(Path(tmpdir))
            trip_ids = set(dfs["trips_supplement.txt"]["trip_id"])
            st_trip_ids = set(dfs["stop_times_supplement.txt"]["trip_id"])
            self.assertTrue(trip_ids.issubset(st_trip_ids))

    def test_routes_supplement_route_type_is_bus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dfs = self._write_and_read(Path(tmpdir))
            routes = dfs["routes_supplement.txt"]
            self.assertTrue((routes["route_type"] == "3").all())

    def test_output_dir_created_if_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "a" / "b" / "tods"
            write_tods_deadhead(
                deadhead_trips=self.deadhead_trips,
                deadhead_stop_times=self.deadhead_stop_times,
                deadhead_stops=self.deadhead_stops,
                shapes=self.shapes,
                gtfs_stops=self.gtfs_stops,
                output_dir=nested,
            )
            self.assertTrue((nested / "trips_supplement.txt").exists())


if __name__ == "__main__":
    unittest.main()
