import logging
import os

import boto3
import geopandas as gpd
import numpy as np
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from gtfsblocks import Feed

TMY_DIR = "./TMY"  # Folder for saving the downloaded TMY data
logger = logging.getLogger(__name__)


def fetch_counties_gdf() -> gpd.DataFrame:
    gdf_county = gpd.read_file(
        "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_20m.zip"
    )
    gdf_county["county_id"] = (
        "G" + gdf_county["STATEFP"] + "0" + gdf_county["COUNTYFP"] + "0"
    )
    return gdf_county


def download_tmy_files(county_ids: list[str]) -> None:
    """
    Download and save TMY weather files for estimating thermal energy demand.

    TMY stands for Typical Meteorological Year, a dataset that provides representative
    hourly weather data for a location over a synthetic year.  Unlike Actual
    Meteorological Year (AMY) files, which reflect the observed conditions in a specific
    calendar year, TMY files are constructed by selecting typical months from multiple
    years of historical records. This approach smooths out unusual extremes and produces
    a “typical” climate profile, making TMY data well-suited for long-term energy
    modeling and system design studies.

    This function downloads TMY files for all the supplied county IDs and saves them to
    TMY_DIR. It returns None.

    Parameters
    ----------
    county_ids : list[str]
        List of US Census County IDs for which to download TMY files
    trips_df : pd.DataFrame
        Trips on selected date and route, including deadhead trips.
    """
    bucket_name = "oedi-data-lake"  # S3 Bucket and path prefix for TMY data
    prefix = (
        "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/"
        "resstock_tmy3_release_1/weather/tmy3/"
    )
    # Create anonymous S3 client
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    if not os.path.exists(TMY_DIR):
        os.makedirs(TMY_DIR, exist_ok=True)

    # Download files for each county
    for county_id in county_ids:
        file_key = f"{prefix}{county_id}_tmy3.csv"
        local_file = os.path.join(TMY_DIR, f"{county_id}.csv")
        if not os.path.isfile(local_file):
            s3.download_file(bucket_name, file_key, local_file)
            print(f"Downloaded: {county_id}.csv")


def load_thermal_lookup_table() -> pd.DataFrame:
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

    return df_temp_energy


def compute_HVAC_energy(
    start_hours: pd.Series,
    end_hours: pd.Series,
    power_array: np.typing.NDArray[np.number],
) -> np.typing.NDArray[np.number]:
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
    power_24 = np.concatenate((power_array, [power_array[0]]))  # wrap for interpolation

    def interp_power(t: np.number) -> float:
        """Linearly interpolate power at fractional hour t."""
        i = int(np.floor(t)) % 24
        frac = t - np.floor(t)
        return float((1 - frac) * power_24[i] + frac * power_24[i + 1])

    energies = []
    for s, e in zip(start_hours, end_hours):
        # sample in small steps for accurate integration
        ts = np.arange(s, e, 0.01)  # 0.01 h = 36 s resolution
        ps = np.array([interp_power(t) for t in ts])
        energy = np.trapz(ps, ts)  # integrate kW over hours → kWh
        energies.append(energy)

    return np.array(energies)


def get_hourly_temperature(
    county_id: str,
    scenario: str,
) -> pd.DataFrame:
    local_file = os.path.join(TMY_DIR, f"{county_id}.csv")
    tmy_df = pd.read_csv(local_file, parse_dates=["date_time"])[
        ["date_time", "Dry Bulb Temperature [°C]"]
    ]
    tmy_df["day_of_year"] = tmy_df["date_time"].dt.day_of_year
    mean_temp_by_day = (
        tmy_df.groupby("day_of_year")
        .agg(avg_temp_C=("Dry Bulb Temperature [°C]", "mean"))
        .reset_index()
    )

    if scenario == "winter":
        # Find the days with the hottest and coldest average temperature
        cold_day: int = mean_temp_by_day[
            mean_temp_by_day.avg_temp_C == mean_temp_by_day.avg_temp_C.min()
        ]["day_of_year"].iloc[0]

        # Grab the hourly profiles for the coldest and hottest days
        df_out = tmy_df[tmy_df["day_of_year"] == cold_day].copy()

    elif scenario == "summer":
        hot_day: int = mean_temp_by_day[
            mean_temp_by_day.avg_temp_C == mean_temp_by_day.avg_temp_C.max()
        ]["day_of_year"].iloc[0]
        df_out = tmy_df[tmy_df["day_of_year"] == hot_day].copy()

    elif scenario == "median":
        median_day: int = mean_temp_by_day.iloc[
            (mean_temp_by_day["avg_temp_C"] - mean_temp_by_day["avg_temp_C"].median())
            .abs()
            .argsort()[:1]
        ]["day_of_year"].iloc[0]

        df_out = tmy_df[tmy_df["day_of_year"] == median_day].copy()
        df_out["Dry Bulb Temperature [°C]"] = df_out["Dry Bulb Temperature [°C]"].round(
            1
        )

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    df_out["hour"] = df_out["date_time"].dt.hour
    return df_out


def add_HVAC_energy(feed: Feed, trips_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add HVAC energy consumption.

    Parameters
    ----------
    feed : gtfsblocks.Feed
        GTFS feed object containing blocks DataFrame.
    trips_df : pd.DataFrame
        Trips on selected date and route, including deadhead trips.

    Returns
    -------
    pd.DataFrame
        Updated trip level energy prediction DataFrame with HVAC energy consumption data
    """
    # Based on gtfs stops data, get counties served
    df_stops = feed.stops

    gdf_stops = gpd.GeoDataFrame(
        df_stops,
        geometry=gpd.points_from_xy(df_stops.stop_lon, df_stops.stop_lat),
        crs=4269,
    )

    gdf_county = fetch_counties_gdf()

    # Make sure that both GDFs use the same CRS
    if gdf_stops.crs != gdf_county.crs:
        gdf_county = gdf_county.to_crs(gdf_stops.crs)

    # Start by joining directly to counties
    gdf_stops = gpd.sjoin(
        gdf_stops,
        gdf_county[["geometry", "county_id"]],
        how="left",
        predicate="intersects",
    )

    # If any county IDs are NA, use sjoin_nearest to map to the nearest county
    na_mask = gdf_stops["county_id"].isna()
    na_stops = gdf_stops[na_mask]
    if not na_stops.empty:
        na_stops = gdf_stops[na_mask].drop(columns=["index_right", "county_id"])
        # Project for distance calculation
        na_stops = na_stops.to_crs("ESRI:102003")
        na_stops = na_stops.sjoin_nearest(
            right=gdf_county[["geometry", "county_id"]].to_crs("ESRI:102003"),
            how="left",
            max_distance=3000,
        )

        if na_stops["county_id"].isna().sum() > 0:
            raise ValueError(
                "One or more stops are not within 3 km of a county boundary. Unable to "
                "add county-level weather data and HVAC impacts."
            )

        stops_final = pd.concat([gdf_stops[~na_mask], na_stops.to_crs("EPSG:4269")])

    else:
        stops_final = gdf_stops

    county_ids = stops_final["county_id"].unique().tolist()

    # Download TMY Weather Data
    download_tmy_files(county_ids)

    df_temp_energy = load_thermal_lookup_table()

    # Get the power tables for different weather scenarios
    thermal_dfs = []
    for county_id in county_ids:
        for scenario in ["summer", "winter", "median"]:
            hourly_temp_df = get_hourly_temperature(county_id, scenario)
            hourly_temp_df["Dry Bulb Temperature [°C]"] = hourly_temp_df[
                "Dry Bulb Temperature [°C]"
            ].round(1)
            hourly_temp_df = hourly_temp_df.merge(
                df_temp_energy,
                left_on="Dry Bulb Temperature [°C]",
                right_on="Temp_C",
                how="left",
            )
            hourly_temp_df["scenario"] = scenario
            hourly_temp_df["county"] = county_id
            thermal_dfs.append(hourly_temp_df)

    thermal_power_vals = (
        pd.concat(thermal_dfs).groupby(["scenario", "hour"])["Power"].mean()
    )

    # Last calculate the trip HVAC energy
    # Get the start and end time for each trip
    df_stop_times = feed.stop_times[
        feed.stop_times["trip_id"].isin(trips_df["trip_id"].unique())
    ].copy()
    df_trip_time = (
        df_stop_times.groupby("trip_id")
        .agg(start_time=("arrival_time", "min"), end_time=("arrival_time", "max"))
        .reset_index()
    )
    start_hours = df_trip_time["start_time"].dt.total_seconds() / 3600
    end_hours = df_trip_time["end_time"].dt.total_seconds() / 3600

    hvac_df_list = []
    for scen, subdf in thermal_power_vals.reset_index().groupby("scenario"):
        scenario_trips = trips_df.copy()
        scenario_trips["scenario"] = scen
        scenario_trips["hvac_energy"] = compute_HVAC_energy(
            start_hours, end_hours, subdf["Power"].to_numpy()
        )
        hvac_df_list.append(scenario_trips)

    all_predictions = pd.concat(hvac_df_list).reset_index(drop=True)
    trips_df_out = trips_df.merge(
        all_predictions[["trip_id", "scenario", "hvac_energy"]],
        on="trip_id",
        how="inner",
    )
    return trips_df_out
