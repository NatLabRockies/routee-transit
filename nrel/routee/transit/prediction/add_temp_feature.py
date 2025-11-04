import os
from typing import Any

import boto3
import geopandas as gpd
import numpy as np
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config


def add_HVAC_energy(feed: Any, trips_df: pd.DataFrame) -> Any:
    """
    Add HVAC energy consumption.

    Parameters
    ----------
    feed : Any
        GTFS feed object containing blocks DataFrame.
    trips_df : pd.DataFrame
        Trips on selected date and route, including deadhead trips.

    Returns
    -------
    pd.DataFrame
        Updated trip level energy prediction DataFrame with HVAC energy consumption data.
    """
    # Based on gtfs stops data, get counties served
    df_stops = feed.stops
    gdf_stops = gpd.GeoDataFrame(
        df_stops,
        geometry=gpd.points_from_xy(df_stops.stop_lon, df_stops.stop_lat),
        crs=4269,
    )
    gdf_county = gpd.read_file(
        "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_20m.zip"
    )
    gdf_county["county_id"] = (
        "G" + gdf_county["STATEFP"] + "0" + gdf_county["COUNTYFP"] + "0"
    )
    gdf_stops = gpd.sjoin(
        gdf_stops,
        gdf_county[["geometry", "county_id"]],
        how="left",
        predicate="intersects",
    )
    county_ids = gdf_stops.county_id.unique()

    # Download TMY Weather Data
    """TMY stands for Typical Meteorological Year, a dataset that provides representative hourly weather 
    data for a location over a synthetic year. 
    Unlike Actual Meteorological Year (AMY) files, which reflect the observed conditions in a specific calendar year, 
    TMY files are constructed by selecting typical months from multiple years of historical records. 
    This approach smooths out unusual extremes and produces a “typical” climate profile, 
    making TMY data well-suited for long-term energy modeling and system design studies.
    """
    bucket_name = "oedi-data-lake"  # S3 Bucket and path prefix for TMY data
    prefix = "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/resstock_tmy3_release_1/weather/tmy3/"
    s3 = boto3.client(
        "s3", config=Config(signature_version=UNSIGNED)
    )  # Create anonymous S3 client
    save_dir = "./TMY"  # Folder for saving the downloaded TMY data
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # Download files for each county
    for county_id in county_ids:
        file_key = f"{prefix}{county_id}_tmy3.csv"
        local_file = os.path.join(save_dir, f"{county_id}.csv")
        if not os.path.isfile(local_file):
            try:
                s3.download_file(bucket_name, file_key, local_file)
                print(f"Downloaded: {county_id}.csv")
            except Exception as e:
                print(f"Failed to download {county_id}.csv: {e}")

    # Create the HVAC + BTMS power lookup table
    temperature_list = [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]  # From literature
    HVAC_power_list = [25, 17, 10, 6, 4, 1, 1, 2, 4, 7, 11]  # From literature
    BTMS_power_list = [
        4.9,
        3.6,
        2.1,
        0.8,
        0.2,
        0.1,
        0.1,
        1.4,
        1.5,
        2.1,
        5.6,
    ]  # From literature
    total_temp_energy_list = [
        HVAC_power_list[i] + BTMS_power_list[i] for i in range(len(HVAC_power_list))
    ]
    # Add two extreme values to make sure we cover all temperature values
    min_temp = -100
    max_temp = 100
    min_temp_power = (-10 - min_temp) * (
        total_temp_energy_list[0] - total_temp_energy_list[1]
    ) / 5 + total_temp_energy_list[0]
    max_temp_power = (max_temp - 40) * (
        total_temp_energy_list[-1] - total_temp_energy_list[-2]
    ) / 5 + total_temp_energy_list[-1]
    # Extend temp and energy list
    temperature_list = [-100] + temperature_list + [100]
    total_temp_energy_list = (
        [min_temp_power] + total_temp_energy_list + [max_temp_power]
    )
    # Define a dataframe to store the information
    df_temp_energy = pd.DataFrame(
        {"Temp_C": temperature_list, "Power": total_temp_energy_list}
    )
    # Fill every integer Temp_C
    df_tmp_fill = pd.DataFrame({"Temp_C": np.arange(-100, 100.1, 0.1)})
    df_temp_energy["Temp_C"] = df_temp_energy["Temp_C"].astype(float).round(1)
    df_tmp_fill["Temp_C"] = df_tmp_fill["Temp_C"].astype(float).round(1)
    df_temp_energy = df_tmp_fill.merge(df_temp_energy, on="Temp_C", how="left")
    # Linear interpolate
    df_temp_energy["Power"] = df_temp_energy["Power"].interpolate(method="linear")
    df_temp_energy["Temp_C"] = df_temp_energy["Temp_C"].astype(float)

    # Get the winter coldest day and summer hottest day power tables
    # Then loop through each county_id to collect the coldest and hottest day and extract the corresponding power consumption data
    list_df_hot = []
    list_df_cold = []
    for county_id in county_ids:
        file_key = f"{prefix}{county_id}_tmy3.csv"
        local_file = os.path.join(save_dir, f"{county_id}.csv")
        df_temp = pd.read_csv(local_file, parse_dates=["date_time"])
        df_temp["date"] = np.repeat(np.arange(1, 366), 24)
        df_temp_group = (
            df_temp.groupby("date")
            .agg(avg_temp_C=("Dry Bulb Temperature [°C]", "mean"))
            .reset_index()
        )
        cold_day = df_temp_group[
            df_temp_group.avg_temp_C == df_temp_group.avg_temp_C.min()
        ]["date"].iloc[0]
        hot_day = df_temp_group[
            df_temp_group.avg_temp_C == df_temp_group.avg_temp_C.max()
        ]["date"].iloc[0]
        # Get the coldest and hottest day data
        df_cold = df_temp[df_temp.date == cold_day].copy()
        df_cold["Dry Bulb Temperature [°C]"] = df_cold[
            "Dry Bulb Temperature [°C]"
        ].round(1)
        df_cold = df_cold.merge(
            df_temp_energy,
            left_on="Dry Bulb Temperature [°C]",
            right_on="Temp_C",
            how="left",
        )
        df_cold["hour"] = np.arange(0, 24)
        df_hot = df_temp[df_temp.date == hot_day].copy()
        df_hot["Dry Bulb Temperature [°C]"] = df_hot["Dry Bulb Temperature [°C]"].round(
            1
        )
        df_hot = df_hot.merge(
            df_temp_energy,
            left_on="Dry Bulb Temperature [°C]",
            right_on="Temp_C",
            how="left",
        )
        df_hot["hour"] = np.arange(0, 24)
        list_df_hot.append(df_hot)
        list_df_cold.append(df_cold)
    # Mergeg data for all counties
    df_hot_all = pd.concat(list_df_hot)
    df_cold_all = pd.concat(list_df_cold)
    # Next lets get the hourly average values for all stations
    df_cold_hourly = df_cold_all.groupby(["hour"]).agg({"Power": "mean"}).reset_index()
    df_hot_hourly = df_hot_all.groupby(["hour"]).agg({"Power": "mean"}).reset_index()

    # Last calculate the trip HVAC energy
    # Get the start and end time for each trip
    df_stop_times = feed.stop_times
    df_stop_times = df_stop_times[
        df_stop_times["trip_id"].isin(trips_df["trip_id"].unique())
    ].copy()
    df_trip_time = (
        df_stop_times.groupby("trip_id")
        .agg(start_time=("arrival_time", "min"), end_time=("arrival_time", "max"))
        .reset_index()
    )
    start_hours = df_trip_time["start_time"].dt.total_seconds() / 3600
    end_hours = df_trip_time["end_time"].dt.total_seconds() / 3600

    def compute_HVAC_energy(start_hours: Any, end_hours: Any, power_array: Any) -> Any:
        """
        Calculate HVAC + BTMS energy consumption between time intervals.

        Args:
            start_hours (array-like): fractional start times in hours
            end_hours (array-like): fractional end times in hours (can be > 24)
            power_array (array-like): hourly average power values [kW] for hours 0–23

        Returns:
            np.ndarray: energy consumption [kWh] for each interval
        """
        power_array = np.asarray(power_array)
        power_24 = np.concatenate(
            (power_array, [power_array[0]])
        )  # wrap for interpolation

        def interp_power(t: Any) -> Any:
            """Linearly interpolate power at fractional hour t."""
            i = int(np.floor(t)) % 24
            frac = t - np.floor(t)
            return (1 - frac) * power_24[i] + frac * power_24[i + 1]

        energies = []
        for s, e in zip(start_hours, end_hours):
            # sample in small steps for accurate integration
            ts = np.arange(s, e, 0.01)  # 0.01 h = 36 s resolution
            ps = np.array([interp_power(t) for t in ts])
            energy = np.trapz(ps, ts)  # integrate kW over hours → kWh
            energies.append(energy)

        return np.array(energies)

    df_trip_time["Winter_HVAC_Energy"] = compute_HVAC_energy(
        start_hours, end_hours, df_cold_hourly["Power"].values
    )
    df_trip_time["Summer_HVAC_Energy"] = compute_HVAC_energy(
        start_hours, end_hours, df_hot_hourly["Power"].values
    )
    # Merge back to trips_df
    trips_df = trips_df.merge(
        df_trip_time[["trip_id", "Winter_HVAC_Energy", "Summer_HVAC_Energy"]],
        on="trip_id",
        how="left",
    )
    trips_df = trips_df[["trip_id", "Winter_HVAC_Energy", "Summer_HVAC_Energy"]]
    return trips_df
