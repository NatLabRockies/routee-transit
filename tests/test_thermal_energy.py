import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from routee.transit.thermal_energy import (
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
                rows.append({"month": month, "day_of_month": day, "hour": hour, "Power": 5.0})
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
        # scenario is always "TMY"
        self.assertTrue((result["scenario"] == "TMY").all())
        # 2 dates × 1 trip = 2 rows
        self.assertEqual(len(result), 2)

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
        self.assertTrue((result["scenario"] == "TMY").all())
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
