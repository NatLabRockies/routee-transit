import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from routee.transit.thermal_energy import (
    _densest_window,
    _select_full_year_window,
    add_HVAC_energy,
    compute_HVAC_energy,
    load_thermal_lookup_table,
)


class TestThermalEnergy(unittest.TestCase):
    def test_load_thermal_lookup_table(self) -> None:
        df = load_thermal_lookup_table()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("Temp_C", df.columns)
        self.assertIn("Power", df.columns)
        # Check some expected values (linear interpolation)
        # At 20C, HVAC power is 1, BTMS power is 0.1 -> 1.1
        val_20 = df[df["Temp_C"] == 20.0]["Power"].values[0]
        self.assertAlmostEqual(val_20, 1.1, places=1)

    def test_compute_HVAC_energy(self) -> None:
        # 0 to 1 hour, constant 10kW power
        start_hours = pd.Series([0.0])
        end_hours = pd.Series([1.0])
        power_array = np.full(24, 10.0)

        energy = compute_HVAC_energy(start_hours, end_hours, power_array)
        # Power is 10kW, time is 1h.
        # Integration uses np.arange(0, 1, 0.01) which stops at 0.99.
        # np.trapezoid on [0, ..., 0.99] with constant 10 gives 9.9.
        self.assertAlmostEqual(energy[0], 9.9, places=1)

    def _make_mock_feed_and_trips(
        self,
    ) -> tuple[MagicMock, pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame]:
        mock_feed = MagicMock()
        mock_feed.stops = pd.DataFrame(
            {"stop_id": ["S1"], "stop_lat": [40.0], "stop_lon": [-105.0]}
        )
        mock_feed.stop_times = pd.DataFrame(
            {
                "trip_id": ["T1", "T1"],
                "arrival_time": [pd.Timedelta(hours=8), pd.Timedelta(hours=9)],
                "stop_id": ["S1", "S1"],
            }
        )
        # Two service dates, both running service_id "S1" → trip "T1"
        mock_feed.get_service_ids_all_dates.return_value = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-15", "2023-07-15"]),
                "service_id": ["S1", "S1"],
                "weekday": ["sunday", "saturday"],
            }
        )
        trips_df = pd.DataFrame({"trip_id": ["T1"], "service_id": ["S1"]})
        mock_county_gdf = gpd.GeoDataFrame(
            {
                "county_id": ["G0800130"],
                "STATEFP": ["08"],
                "COUNTYFP": ["013"],
                "geometry": [Point(-105.0, 40.0).buffer(1.0)],
            },
            crs="EPSG:4269",
        )
        # Synthetic per-day power table: one row per (month, day_of_month, hour)
        # covering just the two dates used in the test (Jan 15, Jul 15)
        rows = []
        for month, day in [(1, 15), (7, 15)]:
            for hour in range(24):
                rows.append(
                    {"month": month, "day_of_month": day, "hour": hour, "Power": 5.0}
                )
        mock_power_by_day = pd.DataFrame(rows)
        return mock_feed, trips_df, mock_county_gdf, mock_power_by_day

    @patch("routee.transit.thermal_energy.fetch_counties_gdf")
    @patch("routee.transit.thermal_energy.download_tmy_files")
    @patch("routee.transit.thermal_energy.load_tmy_power_by_day")
    def test_add_HVAC_energy(
        self,
        mock_load_power: MagicMock,
        mock_download: MagicMock,
        mock_fetch_counties: MagicMock,
    ) -> None:
        mock_feed, trips_df, mock_county_gdf, mock_power_by_day = (
            self._make_mock_feed_and_trips()
        )
        mock_fetch_counties.return_value = mock_county_gdf
        mock_load_power.return_value = mock_power_by_day

        # use a temp directory for output
        output_directory = Path(tempfile.mkdtemp())

        result = add_HVAC_energy(mock_feed, trips_df, output_directory)

        self.assertIn("hvac_energy_kWh", result.columns)
        self.assertIn("scenario", result.columns)
        self.assertIn("date", result.columns)
        self.assertIn("trip_is_within_gtfs_scope", result.columns)
        # scenario is always "TMY"
        self.assertTrue((result["scenario"] == "TMY").all())
        # Default (scale_to_year=False) only returns covered dates: 2 rows
        self.assertEqual(len(result), 2)
        # All rows are within the feed's real scope
        self.assertTrue(result["trip_is_within_gtfs_scope"].all())

    @patch("routee.transit.thermal_energy.fetch_counties_gdf")
    @patch("routee.transit.thermal_energy.download_tmy_files")
    @patch("routee.transit.thermal_energy.load_tmy_power_by_day")
    def test_add_HVAC_energy_no_output_dir(
        self,
        mock_load_power: MagicMock,
        mock_download: MagicMock,
        mock_fetch_counties: MagicMock,
    ) -> None:
        """add_HVAC_energy should work without output_dir using a default cache path."""
        mock_feed, trips_df, mock_county_gdf, mock_power_by_day = (
            self._make_mock_feed_and_trips()
        )
        mock_fetch_counties.return_value = mock_county_gdf
        mock_load_power.return_value = mock_power_by_day

        # Call without specifying output_dir (previously raised an exception)
        result = add_HVAC_energy(mock_feed, trips_df, output_dir=None)

        self.assertIn("hvac_energy_kWh", result.columns)
        self.assertIn("scenario", result.columns)
        self.assertIn("date", result.columns)
        self.assertIn("trip_is_within_gtfs_scope", result.columns)
        self.assertTrue((result["scenario"] == "TMY").all())
        self.assertEqual(len(result), 2)


class TestDateWindowSelection(unittest.TestCase):
    def test_densest_window_picks_dense_cluster_over_outlier(self) -> None:
        # 10 dense dates in Jan 2024 + one far-future outlier in 2030
        dense = pd.date_range("2024-01-01", periods=10, freq="D").tolist()
        sorted_dates = dense + [pd.Timestamp("2030-06-01")]
        sorted_dates.sort()
        start, end = _densest_window(sorted_dates, max_days=365)  # type: ignore[misc]
        # Dense block should sit inside the chosen window
        self.assertLessEqual(start, dense[0])
        self.assertGreaterEqual(end, dense[-1])
        # The 2030 outlier must be outside the window
        self.assertGreater(pd.Timestamp("2030-06-01"), end)

    def test_densest_window_handles_past_outlier(self) -> None:
        # Symmetric case: far-past outlier should be excluded
        dense = pd.date_range("2024-01-01", periods=10, freq="D").tolist()
        sorted_dates = [pd.Timestamp("2018-06-01")] + dense
        sorted_dates.sort()
        start, end = _densest_window(sorted_dates, max_days=365)  # type: ignore[misc]
        self.assertLess(pd.Timestamp("2018-06-01"), start)
        self.assertLessEqual(start, dense[0])

    def test_select_full_year_snaps_to_calendar_year(self) -> None:
        sorted_dates = pd.date_range("2023-03-01", "2023-08-31", freq="D").tolist()
        start, end = _select_full_year_window(sorted_dates, max_days=365)
        self.assertEqual(start, pd.Timestamp("2023-01-01"))
        self.assertEqual(end, pd.Timestamp("2023-12-31"))

    def test_select_full_year_centers_cross_year_feed(self) -> None:
        # Dense block spanning two calendar years → should center on midpoint
        sorted_dates = pd.date_range("2023-10-01", "2024-03-31", freq="D").tolist()
        start, end = _select_full_year_window(sorted_dates, max_days=365)
        self.assertNotEqual(start, pd.Timestamp("2023-01-01"))
        self.assertNotEqual(start, pd.Timestamp("2024-01-01"))
        # Window covers the full dense block
        self.assertLessEqual(start, sorted_dates[0])
        self.assertGreaterEqual(end, sorted_dates[-1])


class TestScaleToYear(unittest.TestCase):
    def _make_mwf_feed_and_trips(
        self,
    ) -> tuple[MagicMock, pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame]:
        """Build a mock feed that serves only Mon/Wed/Fri in early 2023."""
        mock_feed = MagicMock()
        mock_feed.stops = pd.DataFrame(
            {"stop_id": ["S1"], "stop_lat": [40.0], "stop_lon": [-105.0]}
        )
        mock_feed.stop_times = pd.DataFrame(
            {
                "trip_id": ["T1", "T1"],
                "arrival_time": [pd.Timedelta(hours=8), pd.Timedelta(hours=9)],
                "stop_id": ["S1", "S1"],
            }
        )
        # Real service: Mon/Wed/Fri across two weeks of Jan 2023.
        # 2023-01-02 (Mon), 01-04 (Wed), 01-06 (Fri),
        # 2023-01-09 (Mon), 01-11 (Wed), 01-13 (Fri)
        real_dates = pd.to_datetime(
            [
                "2023-01-02",
                "2023-01-04",
                "2023-01-06",
                "2023-01-09",
                "2023-01-11",
                "2023-01-13",
            ]
        )
        weekdays = [d.day_name().lower() for d in real_dates]
        mock_feed.get_service_ids_all_dates.return_value = pd.DataFrame(
            {
                "date": real_dates,
                "service_id": ["S1"] * 6,
                "weekday": weekdays,
            }
        )
        # Typical service: only Mon/Wed/Fri have any service_id
        mock_feed.get_typical_service_by_weekday.return_value = pd.DataFrame(
            {
                "service_id": [frozenset({"S1"})] * 3,
                "n_dates": [2, 2, 2],
            },
            index=pd.Index(["monday", "wednesday", "friday"], name="weekday"),
        )
        trips_df = pd.DataFrame({"trip_id": ["T1"], "service_id": ["S1"]})
        mock_county_gdf = gpd.GeoDataFrame(
            {
                "county_id": ["G0800130"],
                "STATEFP": ["08"],
                "COUNTYFP": ["013"],
                "geometry": [Point(-105.0, 40.0).buffer(1.0)],
            },
            crs="EPSG:4269",
        )
        # Synthetic per-day power table covering every (month, day_of_month)
        # in a non-leap year.
        rows = []
        for ts in pd.date_range("2023-01-01", "2023-12-31", freq="D"):
            for hour in range(24):
                rows.append(
                    {
                        "month": ts.month,
                        "day_of_month": ts.day,
                        "hour": hour,
                        "Power": 5.0,
                    }
                )
        mock_power_by_day = pd.DataFrame(rows)
        return mock_feed, trips_df, mock_county_gdf, mock_power_by_day

    @patch("routee.transit.thermal_energy.fetch_counties_gdf")
    @patch("routee.transit.thermal_energy.download_tmy_files")
    @patch("routee.transit.thermal_energy.load_tmy_power_by_day")
    def test_scale_to_year_projects_only_to_feed_weekdays(
        self,
        mock_load_power: MagicMock,
        mock_download: MagicMock,
        mock_fetch_counties: MagicMock,
    ) -> None:
        mock_feed, trips_df, mock_county_gdf, mock_power_by_day = (
            self._make_mwf_feed_and_trips()
        )
        mock_fetch_counties.return_value = mock_county_gdf
        mock_load_power.return_value = mock_power_by_day

        result = add_HVAC_energy(
            mock_feed, trips_df, output_dir=None, scale_to_year=True
        )

        # Window snaps to 2023 calendar year (all real dates in 2023).
        # 2023-01-01 is a Sunday → no service projected, so the first
        # produced date is the first Monday (2023-01-02); likewise the
        # last day is the last Friday on/before Dec 31 (2023-12-29).
        self.assertGreaterEqual(result["date"].min(), pd.Timestamp("2023-01-01"))
        self.assertLessEqual(result["date"].max(), pd.Timestamp("2023-12-31"))
        # Every produced date must be a Mon/Wed/Fri (the feed's only service days)
        weekday_nums = result["date"].dt.weekday.unique()
        # 0=Mon, 2=Wed, 4=Fri
        self.assertEqual(set(weekday_nums.tolist()), {0, 2, 4})
        # Real (covered) dates have flag True, synthesized ones False
        real_dates = {
            pd.Timestamp(d)
            for d in [
                "2023-01-02",
                "2023-01-04",
                "2023-01-06",
                "2023-01-09",
                "2023-01-11",
                "2023-01-13",
            ]
        }
        for _, row in result.iterrows():
            if row["date"] in real_dates:
                self.assertTrue(row["trip_is_within_gtfs_scope"])
            else:
                self.assertFalse(row["trip_is_within_gtfs_scope"])

    @patch("routee.transit.thermal_energy.fetch_counties_gdf")
    @patch("routee.transit.thermal_energy.download_tmy_files")
    @patch("routee.transit.thermal_energy.load_tmy_power_by_day")
    def test_scale_to_year_false_leaves_output_unchanged(
        self,
        mock_load_power: MagicMock,
        mock_download: MagicMock,
        mock_fetch_counties: MagicMock,
    ) -> None:
        mock_feed, trips_df, mock_county_gdf, mock_power_by_day = (
            self._make_mwf_feed_and_trips()
        )
        mock_fetch_counties.return_value = mock_county_gdf
        mock_load_power.return_value = mock_power_by_day

        result = add_HVAC_energy(mock_feed, trips_df, output_dir=None)

        # Only the 6 real dates appear, all flagged True
        self.assertEqual(len(result), 6)
        self.assertTrue(result["trip_is_within_gtfs_scope"].all())


if __name__ == "__main__":
    unittest.main()
