import csv
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from routee.transit.depot_deadhead import (
    create_depot_deadhead_stops,
    create_depot_deadhead_trips,
    infer_depot_trip_endpoints,
)
from routee.transit.ntd import (
    _compute_token_idf,
    _load_ntd_agencies,
    match_agency_to_ntd,
)


class TestCreateDepotDeadheadTrips(unittest.TestCase):
    def setUp(self) -> None:
        # Create sample trips and stop_times dataframes
        self.trips_df = pd.DataFrame(
            {
                "trip_id": ["trip1", "trip2", "trip3", "trip4"],
                "block_id": ["block1", "block1", "block2", "block2"],
                "service_id": ["service1", "service1", "service2", "service2"],
                "agency_id": ["agency1", "agency1", "agency2", "agency2"],
            }
        )

        self.stop_times_df = pd.DataFrame(
            {
                "trip_id": ["trip1", "trip1", "trip2", "trip2", "trip3", "trip4"],
                "arrival_time": [
                    pd.Timedelta(hours=8),
                    pd.Timedelta(hours=8, minutes=30),
                    pd.Timedelta(hours=9),
                    pd.Timedelta(hours=9, minutes=30),
                    pd.Timedelta(hours=10),
                    pd.Timedelta(hours=11),
                ],
            }
        )

    def test_create_depot_deadhead_trips_basic(self) -> None:
        result = create_depot_deadhead_trips(self.trips_df, self.stop_times_df)

        # Should create 2 trips per block (pull-out and pull-in)
        self.assertEqual(len(result), 4)

        # Check pull-out trips
        pullout_trips = result[result["trip_type"] == "pull-out"]
        self.assertEqual(len(pullout_trips), 2)
        self.assertTrue(all(pullout_trips["trip_id"].str.startswith("depot_to_")))

        # Check pull-in trips
        pullin_trips = result[result["trip_type"] == "pull-in"]
        self.assertEqual(len(pullin_trips), 2)
        self.assertTrue(all(pullin_trips["trip_id"].str.endswith("_to_depot")))

    def test_create_depot_deadhead_trips_columns(self) -> None:
        result = create_depot_deadhead_trips(self.trips_df, self.stop_times_df)

        required_columns = [
            "trip_id",
            "trip_type",
            "route_id",
            "service_id",
            "block_id",
            "shape_id",
            "route_short_name",
            "route_type",
            "route_desc",
            "agency_id",
        ]

        for col in required_columns:
            self.assertIn(col, result.columns)

    def test_create_depot_deadhead_trips_route_type(self) -> None:
        result = create_depot_deadhead_trips(self.trips_df, self.stop_times_df)

        # All route types should be 3 (bus)
        self.assertTrue(all(result["route_type"] == 3))

    def test_agency_id_propagated_from_first_and_last_trip(self) -> None:
        result = create_depot_deadhead_trips(self.trips_df, self.stop_times_df)

        # pull-out trips should inherit agency_id from the first revenue trip of
        # the block; pull-in trips from the last revenue trip.
        self.assertIn("agency_id", result.columns)

        for _, row in result.iterrows():
            block = row["block_id"]
            expected = self.trips_df.loc[
                self.trips_df["block_id"] == block, "agency_id"
            ].iloc[0]
            self.assertEqual(
                row["agency_id"],
                expected,
                msg=f"agency_id mismatch for {row['trip_id']}",
            )

    def test_agency_id_none_when_absent_from_trips(self) -> None:
        trips_no_agency = self.trips_df.drop(columns=["agency_id"])
        result = create_depot_deadhead_trips(trips_no_agency, self.stop_times_df)

        # agency_id column should still exist but be None/NaN when not present
        # in the source trips (get() fallback returns None).
        self.assertIn("agency_id", result.columns)
        self.assertTrue(result["agency_id"].isna().all())

    def test_create_depot_deadhead_trips_with_existing_deadhead(self) -> None:
        # Add a between-trip deadhead to trips
        trips_with_deadhead = self.trips_df.copy()
        trips_with_deadhead["from_trip"] = [None, "trip1", None, None]

        result = create_depot_deadhead_trips(trips_with_deadhead, self.stop_times_df)

        # Should still create 4 depot deadhead trips (2 per block)
        self.assertEqual(len(result), 4)


class TestInferDepotTripEndpoints(unittest.TestCase):
    def setUp(self) -> None:
        # Create sample trips
        self.trips_df = pd.DataFrame(
            {
                "trip_id": ["trip1", "trip2"],
                "block_id": ["block1", "block1"],
            }
        )

        # Create mock feed
        self.feed = MagicMock()
        self.feed.stop_times = pd.DataFrame(
            {
                "trip_id": ["trip1", "trip1", "trip2", "trip2"],
                "stop_id": ["stop1", "stop2", "stop3", "stop4"],
                "arrival_time": [
                    pd.Timedelta(hours=8),
                    pd.Timedelta(hours=8, minutes=30),
                    pd.Timedelta(hours=9),
                    pd.Timedelta(hours=9, minutes=30),
                ],
            }
        )
        self.feed.stops = pd.DataFrame(
            {
                "stop_id": ["stop1", "stop2", "stop3", "stop4"],
                "stop_lat": [39.0, 39.1, 39.2, 39.3],
                "stop_lon": [-105.0, -105.1, -105.2, -105.3],
            }
        )

        # Create an in-memory depot GeoDataFrame (replaces old shapefile approach)
        self.depots_gdf = gpd.GeoDataFrame(
            {
                "Facility Type": [
                    "General Purpose Maintenance Facility/Depot",
                    "Maintenance Facility (Service and Inspection)",
                ],
                "depot_priority": [0, 2],
                "geometry": [Point(-105.05, 39.05), Point(-105.25, 39.25)],
            },
            crs="EPSG:4326",
        )

    def test_infer_depot_trip_endpoints_returns_geodataframes(self) -> None:
        first_stops, last_stops, depots_df = infer_depot_trip_endpoints(
            self.trips_df, self.feed, self.depots_gdf
        )

        self.assertIsInstance(first_stops, gpd.GeoDataFrame)
        self.assertIsInstance(last_stops, gpd.GeoDataFrame)

    def test_infer_depot_trip_endpoints_columns(self) -> None:
        first_stops, last_stops, depots_df = infer_depot_trip_endpoints(
            self.trips_df, self.feed, self.depots_gdf
        )

        # Check required columns
        for gdf in [first_stops, last_stops]:
            self.assertIn("block_id", gdf.columns)
            self.assertIn("geometry_origin", gdf.columns)
            self.assertIn("geometry_destination", gdf.columns)

    def test_infer_depot_trip_endpoints_geometry_types(self) -> None:
        first_stops, last_stops, depots_df = infer_depot_trip_endpoints(
            self.trips_df, self.feed, self.depots_gdf
        )

        # All geometries should be Points
        for gdf in [first_stops, last_stops]:
            for geom in gdf["geometry_origin"]:
                self.assertEqual(geom.geom_type, "Point")
            for geom in gdf["geometry_destination"]:
                self.assertEqual(geom.geom_type, "Point")

    def test_infer_depot_trip_endpoints_priority_selects_highest_tier(self) -> None:
        """When priority-0 depots exist they should be preferred over priority-2."""
        first_stops, last_stops, _ = infer_depot_trip_endpoints(
            self.trips_df, self.feed, self.depots_gdf
        )
        # The priority-0 depot is at (-105.05, 39.05) — nearest to the first stop
        # at (-105.0, 39.0).  Confirm its x-coordinate is used.
        origin_geom = first_stops.iloc[0]["geometry_origin"]
        self.assertAlmostEqual(origin_geom.x, -105.05, places=4)

    def test_infer_depot_trip_endpoints_crs(self) -> None:
        first_stops, last_stops, depots_df = infer_depot_trip_endpoints(
            self.trips_df, self.feed, self.depots_gdf
        )

        # Both should be in EPSG:4326
        self.assertEqual(first_stops.crs.to_string(), "EPSG:4326")
        self.assertEqual(last_stops.crs.to_string(), "EPSG:4326")

    def test_infer_depot_trip_endpoints_ntd_metadata_columns(self) -> None:
        """NTD metadata columns are propagated to both stop GDFs."""
        depots_gdf = gpd.GeoDataFrame(
            {
                "Facility Type": ["General Purpose Maintenance Facility/Depot"],
                "Facility Name": ["Central Bus Yard"],
                "NTD ID": ["00001"],
                "Agency Name": ["King County"],
                "depot_priority": [0],
                "geometry": [Point(-105.05, 39.05)],
            },
            crs="EPSG:4326",
        )
        first_stops, last_stops, _ = infer_depot_trip_endpoints(
            self.trips_df, self.feed, depots_gdf
        )
        for gdf in [first_stops, last_stops]:
            self.assertIn("depot_ntd_id", gdf.columns)
            self.assertIn("depot_agency_name", gdf.columns)
            self.assertIn("depot_facility_name", gdf.columns)
            self.assertIn("depot_facility_type", gdf.columns)
            self.assertEqual(gdf.iloc[0]["depot_ntd_id"], "00001")
            self.assertEqual(gdf.iloc[0]["depot_facility_name"], "Central Bus Yard")


class TestCreateDepotDeadheadStops(unittest.TestCase):
    def setUp(self) -> None:
        # Create sample GeoDataFrames with proper geometry column
        self.first_stops_gdf = gpd.GeoDataFrame(
            {
                "block_id": ["block1"],
                "stop_id": ["stop_gtfs_1"],  # existing GTFS stop at first revenue stop
                "nearest_depot_idx": [42],
                "arrival_time": [pd.Timedelta(hours=8)],
                "geometry_origin": [Point(-105.0, 39.0)],
                "geometry_destination": [Point(-105.1, 39.1)],
            },
            geometry="geometry_destination",
            crs="EPSG:4326",
        )

        self.last_stops_gdf = gpd.GeoDataFrame(
            {
                "block_id": ["block1"],
                "stop_id": ["stop_gtfs_2"],  # existing GTFS stop at last revenue stop
                "nearest_depot_idx": [42],
                "departure_time": [pd.Timedelta(hours=17)],
                "geometry_origin": [Point(-105.2, 39.2)],
                "geometry_destination": [Point(-105.3, 39.3)],
            },
            geometry="geometry_origin",
            crs="EPSG:4326",
        )

        self.deadhead_trips = pd.DataFrame(
            {
                "trip_id": ["depot_to_trip1", "trip2_to_depot"],
                "trip_type": ["pull-out", "pull-in"],
                "block_id": ["block1", "block1"],
            }
        )

    def test_create_depot_deadhead_stops_returns_dataframes(self) -> None:
        stop_times, stops = create_depot_deadhead_stops(
            self.first_stops_gdf, self.last_stops_gdf, self.deadhead_trips
        )

        self.assertIsInstance(stop_times, pd.DataFrame)
        self.assertIsInstance(stops, pd.DataFrame)

    def test_create_depot_deadhead_stops_columns(self) -> None:
        stop_times, stops = create_depot_deadhead_stops(
            self.first_stops_gdf, self.last_stops_gdf, self.deadhead_trips
        )

        # Check stop_times columns
        required_stop_times_cols = [
            "trip_id",
            "stop_sequence",
            "arrival_time",
            "stop_id",
            "departure_time",
            "shape_dist_traveled",
        ]
        for col in required_stop_times_cols:
            self.assertIn(col, stop_times.columns)

        # Check stops columns
        required_stops_cols = ["stop_id", "stop_lat", "stop_lon"]
        for col in required_stops_cols:
            self.assertIn(col, stops.columns)

    def test_create_depot_deadhead_stops_count(self) -> None:
        stop_times, stops = create_depot_deadhead_stops(
            self.first_stops_gdf, self.last_stops_gdf, self.deadhead_trips
        )

        # Each deadhead trip should have 2 stop_times rows (origin and destination)
        # 2 trips * 2 stops = 4 total
        self.assertEqual(len(stop_times), 4)
        # Only the depot stop is new; revenue endpoints already exist in GTFS.
        # Both pull-out and pull-in share the same depot_block1 stop -> 1 unique row.
        self.assertEqual(len(stops), 1)

    def test_create_depot_deadhead_stops_sequences(self) -> None:
        stop_times, stops = create_depot_deadhead_stops(
            self.first_stops_gdf, self.last_stops_gdf, self.deadhead_trips
        )

        # Check stop sequences are 1 and 2 for each trip
        for trip_id in self.deadhead_trips["trip_id"]:
            trip_stops = stop_times[stop_times["trip_id"] == trip_id]
            sequences = sorted(trip_stops["stop_sequence"].tolist())
            self.assertEqual(sequences, [1, 2])

    def test_create_depot_deadhead_stops_stop_ids(self) -> None:
        stop_times, stops = create_depot_deadhead_stops(
            self.first_stops_gdf, self.last_stops_gdf, self.deadhead_trips
        )

        # The only new stop is the depot, keyed as "depot_{nearest_depot_idx}".
        self.assertEqual(list(stops["stop_id"]), ["depot_42"])

        # stop_times should reference depot_42 for depot endpoints and
        # real GTFS stop IDs for revenue endpoints.
        all_stop_ids = set(stop_times["stop_id"].tolist())
        self.assertIn("depot_42", all_stop_ids)
        self.assertIn("stop_gtfs_1", all_stop_ids)  # first revenue stop
        self.assertIn("stop_gtfs_2", all_stop_ids)  # last revenue stop

    def test_create_depot_deadhead_stops_coordinates(self) -> None:
        stop_times, stops = create_depot_deadhead_stops(
            self.first_stops_gdf, self.last_stops_gdf, self.deadhead_trips
        )

        # Check that coordinates are valid
        self.assertTrue(all(stops["stop_lat"].between(-90, 90)))
        self.assertTrue(all(stops["stop_lon"].between(-180, 180)))

    def test_create_depot_deadhead_stops_name_from_facility(self) -> None:
        """stop_name is set to depot_facility_name when that column is present."""
        first_stops = self.first_stops_gdf.copy()
        last_stops = self.last_stops_gdf.copy()
        first_stops["depot_facility_name"] = "Central Bus Yard"
        last_stops["depot_facility_name"] = "Central Bus Yard"

        _, stops = create_depot_deadhead_stops(
            first_stops, last_stops, self.deadhead_trips
        )

        self.assertIn("stop_name", stops.columns)
        self.assertEqual(stops.iloc[0]["stop_name"], "Central Bus Yard")


class TestMatchAgencyToNtd(unittest.TestCase):
    # Seattle city center; used as a proxy location for King County Metro.
    seattle_lat = 47.527344
    seattle_lon = -122.146266
    sound_lat = 47.536243
    sound_lon = -122.180237

    def test_match_agency_to_ntd_king_county_name(self) -> None:
        result = match_agency_to_ntd(
            agency_name="King County",
            lat=self.seattle_lat,
            lon=self.seattle_lon,
        )
        self.assertEqual(result["NTD_ID"], "00001")

    def test_match_agency_to_ntd_metro_transit_name(self) -> None:
        result = match_agency_to_ntd(
            agency_name="Metro Transit",
            lat=self.seattle_lat,
            lon=self.seattle_lon,
            agency_id="1",
        )
        self.assertEqual(result["NTD_ID"], "00001")

    def test_match_agency_to_ntd_king_county_metro_transit_name(self) -> None:
        result = match_agency_to_ntd(
            agency_name="King County Metro Transit",
            lat=self.seattle_lat,
            lon=self.seattle_lon,
        )
        self.assertEqual(result["NTD_ID"], "00001")

    def test_match_agency_to_ntd_king_county_metro_name(self) -> None:
        result = match_agency_to_ntd(
            agency_name="King County Metro",
            lat=self.seattle_lat,
            lon=self.seattle_lon,
        )
        self.assertEqual(result["NTD_ID"], "00001")

    def test_match_agency_to_ntd_sound_transit_name(self) -> None:
        result = match_agency_to_ntd(
            agency_name="Sound Transit",
            lat=self.sound_lat,
            lon=self.sound_lon,
        )
        self.assertEqual(result["NTD_ID"], "00040")

    def test_match_agency_to_ntd_soun_transt_name(self) -> None:
        result = match_agency_to_ntd(
            agency_name="Soun Transt",
            lat=self.sound_lat,
            lon=self.sound_lon,
        )
        self.assertEqual(result["NTD_ID"], "00040")

    # Frederick MD
    def test_match_agency_to_ntd_frederick(self) -> None:
        result = match_agency_to_ntd(
            agency_name="Transit Services of Frederick County",
            lat=39.489822,
            lon=-77.488062,
        )
        self.assertEqual(result["NTD_ID"], "30072")

    # Mountain Line, Missoula, MT
    def test_match_agency_to_ntd_missoula(self) -> None:
        result = match_agency_to_ntd(
            agency_name="Missoula Urban Transportation District",
            lat=46.872166,
            lon=-113.975243,
        )
        self.assertEqual(result["NTD_ID"], "80009")

    # RTD, Denver
    def test_match_agency_to_ntd_denver(self) -> None:
        result = match_agency_to_ntd(
            agency_name="Regional Transportation District (RTD)",
            lat=39.82,
            lon=-105.1,
        )
        self.assertEqual(result["NTD_ID"], "80006")

    def test_token_frequency_downweights_common_words(self) -> None:
        agencies = _load_ntd_agencies()
        token_idf = _compute_token_idf(agencies)

        # Common tokens should carry less signal than rare identifying tokens.
        self.assertLess(token_idf["transit"], token_idf["muckleshoot"])
        self.assertLess(token_idf["transit"], token_idf["king"])


class TestNTDMatchingFromCSV(unittest.TestCase):
    """Parametrized test that checks all labelled WA agency matches from CSV."""

    _test_csv = Path(__file__).parent / "ntd-agencies-test.csv"

    def test_all_wa_agencies(self) -> None:
        failures: list[str] = []
        with open(self._test_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                expected_id = row["ntd_agency_id"].strip()
                if not expected_id:
                    continue
                expected_id = expected_id.zfill(5)
                agency_name = row["agency_name"].strip()
                lat = float(row["center_latitude"])
                lon = float(row["center_longitude"])
                agency_id = row["agency_id"].strip() or None

                result = match_agency_to_ntd(
                    agency_name=agency_name,
                    lat=lat,
                    lon=lon,
                    agency_id=agency_id,
                )
                if result["NTD_ID"] != expected_id:
                    failures.append(
                        f"{agency_name}: expected {expected_id}, got {result['NTD_ID']}"
                    )

        if failures:
            self.fail(f"{len(failures)} NTD match failures:\n" + "\n".join(failures))


class TestNTDNonMatchesFromCSV(unittest.TestCase):
    """Agencies that must NOT return a known-wrong NTD match.

    ``ntd-agencies-negative-test.csv`` lists agencies where the matcher
    previously produced an incorrect NTD ID — either because the agency is not
    in the NTD at all (private operators, campus/intercity carriers) or because
    a differently-named nearby agency was picked. The ``ntd_agency_id`` column
    records that *wrong* id. A correct matcher must either reject the agency or
    return a different (correct) id, so we assert the wrong id is never returned.
    """

    _negative_csv = Path(__file__).parent / "ntd-agencies-negative-test.csv"

    def test_no_known_wrong_matches(self) -> None:
        failures: list[str] = []
        with open(self._negative_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                wrong_id = row["ntd_agency_id"].strip().zfill(5)
                agency_name = row["agency_name"].strip()
                lat = float(row["center_latitude"])
                lon = float(row["center_longitude"])
                agency_id = row["agency_id"].strip() or None

                try:
                    result = match_agency_to_ntd(
                        agency_name=agency_name,
                        lat=lat,
                        lon=lon,
                        agency_id=agency_id,
                    )
                except ValueError:
                    # Rejected as "not in NTD" — acceptable.
                    continue

                if result["NTD_ID"] == wrong_id:
                    failures.append(
                        f"{agency_name}: returned known-wrong id {wrong_id} "
                        f"('{result['Agency_Name']}')"
                    )

        if failures:
            self.fail(
                f"{len(failures)} known-wrong NTD matches:\n" + "\n".join(failures)
            )


if __name__ == "__main__":
    unittest.main()
