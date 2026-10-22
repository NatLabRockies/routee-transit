import logging
from pathlib import Path
from typing import cast

import boto3
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from botocore import UNSIGNED
from botocore.config import Config
from gtfsblocks import Feed

from routee.transit.results import (
    _densest_window as _densest_window,
)
from routee.transit.results import (
    _select_full_year_window as _select_full_year_window,
)
from routee.transit.results import (
    build_trip_dates,
)

logger = logging.getLogger(__name__)

COUNTY_SHAPEFILE_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_20m.zip"
)
DEFAULT_CACHE_DIR = Path.home() / "cache" / "routee-transit"


def fetch_counties_gdf(cache_dir: Path | None = None) -> gpd.GeoDataFrame:
    """Load US county boundaries, downloading the Census archive only once.

    Parameters
    ----------
    cache_dir : Path or None
        Directory holding the cached shapefile archive. Defaults to
        ``~/cache/routee-transit/counties``.

    Returns
    -------
    gpd.GeoDataFrame
        County polygons with a ``county_id`` column matching the naming scheme
        used by the TMY weather files.
    """
    cache_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR / "counties"
    archive = cache_dir / COUNTY_SHAPEFILE_URL.rsplit("/", 1)[-1]
    if not archive.is_file():
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading county boundaries from %s", COUNTY_SHAPEFILE_URL)
        response = requests.get(COUNTY_SHAPEFILE_URL, timeout=120)
        response.raise_for_status()
        # Write aside and rename so an interrupted download is never cached.
        partial = archive.with_suffix(".part")
        partial.write_bytes(response.content)
        partial.replace(archive)

    gdf_county = gpd.read_file(archive)
    gdf_county["county_id"] = (
        "G" + gdf_county["STATEFP"] + "0" + gdf_county["COUNTYFP"] + "0"
    )
    return gdf_county


def download_tmy_files(county_ids: list[str], tmy_dir: Path) -> None:
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
    `tmy_dir`. It returns None.

    Parameters
    ----------
    county_ids : list[str]
        List of US Census County IDs for which to download TMY files.
    tmy_dir : Path
        Directory where downloaded TMY CSV files are saved.
    """
    bucket_name = "oedi-data-lake"  # S3 Bucket and path prefix for TMY data
    prefix = (
        "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/"
        "resstock_tmy3_release_1/weather/tmy3/"
    )
    # Create anonymous S3 client
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    if not tmy_dir.exists():
        tmy_dir.mkdir(parents=True, exist_ok=True)

    # Download files for each county
    for county_id in county_ids:
        file_key = f"{prefix}{county_id}_tmy3.csv"
        local_file = tmy_dir / f"{county_id}.csv"
        if not local_file.is_file():
            s3.download_file(bucket_name, file_key, str(local_file))
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

    The hourly power values define a piecewise-linear profile that wraps at
    midnight, so each interval's energy is evaluated from the profile's exact
    antiderivative rather than by numerical sampling.

    Args:
        start_hours (array-like): fractional start times in hours
        end_hours (array-like): fractional end times in hours (can be > 24)
        power_array (array-like): hourly average power values [kW] for hours 0–23

    Returns:
        np.ndarray: energy consumption [kWh] for each interval
    """
    knots = np.asarray(power_array, dtype=float)
    knots = np.concatenate((knots, [knots[0]]))  # wrap midnight for interpolation
    # Energy accumulated from midnight to each whole hour, plus a full day total.
    cumulative = np.concatenate(([0.0], np.cumsum((knots[:-1] + knots[1:]) / 2)))

    def integral(hours: pd.Series) -> np.typing.NDArray[np.float64]:
        """Energy consumed from midnight of the first day through ``hours``."""
        days, hour = np.divmod(np.asarray(hours, dtype=float), 24.0)
        index = np.floor(hour).astype(int)
        fraction = hour - index
        # Area of the partial trapezoid between hour ``index`` and ``hour``.
        partial = fraction * (
            knots[index] + 0.5 * fraction * (knots[index + 1] - knots[index])
        )
        return cast(
            np.typing.NDArray[np.float64],
            days * cumulative[24] + cumulative[index] + partial,
        )

    return integral(end_hours) - integral(start_hours)


def load_tmy_power_by_day(
    county_id: str,
    tmy_dir: Path,
    df_temp_energy: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load the full TMY temperature profile for a county and convert to power values.

    Parameters
    ----------
    county_id : str
        US Census county identifier.
    tmy_dir : Path
        Directory containing downloaded TMY CSV files.
    df_temp_energy : pd.DataFrame
        Lookup table mapping temperature (Temp_C) to power (Power) in kW,
        as returned by load_thermal_lookup_table().

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: month, day_of_month, hour, Power.
        One row per hour of the synthetic TMY year (8760 rows).
    """
    local_file = tmy_dir / f"{county_id}.csv"
    tmy_df = pd.read_csv(local_file, parse_dates=["date_time"])[
        ["date_time", "Dry Bulb Temperature [°C]"]
    ]
    tmy_df["month"] = tmy_df["date_time"].dt.month
    tmy_df["day_of_month"] = tmy_df["date_time"].dt.day
    tmy_df["hour"] = tmy_df["date_time"].dt.hour
    tmy_df["Dry Bulb Temperature [°C]"] = tmy_df["Dry Bulb Temperature [°C]"].round(1)
    tmy_df = tmy_df.merge(
        df_temp_energy,
        left_on="Dry Bulb Temperature [°C]",
        right_on="Temp_C",
        how="left",
    )
    return tmy_df[["month", "day_of_month", "hour", "Power"]]


def add_HVAC_energy(
    feed: Feed,
    trips_df: pd.DataFrame,
    output_dir: Path | None = None,
    max_days: int = 365,
    service_date: pd.Timestamp | None = None,
    scale_to_year: bool = False,
) -> pd.DataFrame:
    """
    Add HVAC energy consumption for each calendar day covered by the feed.

    Uses Typical Meteorological Year (TMY) temperature data to compute hourly
    HVAC+BTMS power demand for every day the feed is active (up to ``max_days``
    days).  Each trip is replicated once per calendar day on which it runs, with
    ``hvac_energy_kWh`` reflecting the temperature conditions for that specific day.

    Parameters
    ----------
    feed : gtfsblocks.Feed
        GTFS feed object containing blocks DataFrame.
    trips_df : pd.DataFrame
        Trips on selected date and route, including deadhead trips.
        Must contain ``trip_id`` and ``service_id`` columns.
    output_dir : Path or None
        Directory used to store downloaded TMY weather files (in a ``TMY/``
        subdirectory) and county boundaries (in a ``counties/`` subdirectory).
        If None, defaults to ``~/cache/routee-transit``.
    max_days : int, default=365
        Width in calendar days of the window to model.  The algorithm selects the
        ``max_days``-wide window that contains the greatest number of service dates,
        so that it is robust to feeds with sparse outlier dates far in the past or
        future.  Ignored when ``service_date`` is provided.
    service_date : pd.Timestamp or None
        When set, only this single calendar date is modeled for HVAC.  This
        should be supplied when the predictor was run with a specific ``date``
        filter so that the output contains exactly one row per trip rather than
        one row per (trip, day-in-feed).
    scale_to_year : bool, default=False
        When True, project the feed's typical weekday service patterns onto every
        uncovered date in a full ``max_days``-wide window so the output spans an
        entire year regardless of feed coverage.  Synthesized dates are flagged
        with ``trip_is_within_gtfs_scope=False``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``trip_id``, ``date``, ``scenario``,
        ``hvac_energy_kWh``, and ``trip_is_within_gtfs_scope``.  One row per
        (trip, calendar day) combination.  ``scenario`` is always ``"TMY"``.
        ``trip_is_within_gtfs_scope`` is True when the date is part of the
        feed's actual calendar, False when synthesized via ``scale_to_year``.
    """
    cache_root = output_dir if output_dir is not None else DEFAULT_CACHE_DIR
    tmy_dir = cache_root / "TMY"

    # Based on gtfs stops data, get counties served
    df_stops = feed.stops

    gdf_stops = gpd.GeoDataFrame(
        df_stops,
        geometry=gpd.points_from_xy(df_stops.stop_lon, df_stops.stop_lat),
        crs=4269,
    )

    gdf_county = fetch_counties_gdf(cache_root / "counties")

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
    download_tmy_files(county_ids, tmy_dir)

    df_temp_energy = load_thermal_lookup_table()

    # Build a (month, day_of_month, hour) → power table averaged across all counties
    # TODO: use allow shape-dependent TMY data choice
    county_power_dfs = [
        load_tmy_power_by_day(county_id, tmy_dir, df_temp_energy)
        for county_id in county_ids
    ]
    avg_power = (
        pd.concat(county_power_dfs)
        .groupby(["month", "day_of_month", "hour"])["Power"]
        .mean()
        .reset_index()
    )

    # Build lookup: (month, day_of_month) → np.array[24] of hourly power values
    power_lookup = {}
    for key, grp in avg_power.groupby(["month", "day_of_month"]):
        month_key, day_key = cast(tuple[int, int], key)
        power_lookup[(month_key, day_key)] = grp.sort_values("hour")["Power"].to_numpy()

    date_trips = build_trip_dates(
        feed,
        trips_df,
        service_date=service_date,
        scale_to_year=scale_to_year,
        max_days=max_days,
    )

    # Load trip start/end fractional hours from stop_times
    df_stop_times = feed.stop_times[
        feed.stop_times["trip_id"].isin(trips_df["trip_id"].unique())
    ].copy()
    trip_hours = (
        df_stop_times.groupby("trip_id")
        .agg(start_time=("arrival_time", "min"), end_time=("arrival_time", "max"))
        .reset_index()
    )
    trip_hours["start_hour"] = trip_hours["start_time"].dt.total_seconds() / 3600
    trip_hours["end_hour"] = trip_hours["end_time"].dt.total_seconds() / 3600

    date_trips = date_trips.merge(
        trip_hours[["trip_id", "start_hour", "end_hour"]], on="trip_id"
    )

    # Add (month, day_of_month) columns for power lookup
    date_trips["month"] = date_trips["date"].dt.month
    date_trips["day_of_month"] = date_trips["date"].dt.day
    # Feb 29 → Feb 28 (TMY is a non-leap synthetic year)
    feb29_mask = (date_trips["month"] == 2) & (date_trips["day_of_month"] == 29)
    date_trips.loc[feb29_mask, "day_of_month"] = 28

    # Compute HVAC energy for each unique (month, day_of_month) power profile
    result_parts: list[pd.DataFrame] = []
    for key, grp in date_trips.groupby(["month", "day_of_month"], sort=False):
        month_key, day_key = cast(tuple[int, int], key)
        power_array = power_lookup[(month_key, day_key)]
        grp_reset = grp.reset_index(drop=True)
        hvac_vals = compute_HVAC_energy(
            grp_reset["start_hour"], grp_reset["end_hour"], power_array
        )
        part = grp_reset[["trip_id", "date", "trip_is_within_gtfs_scope"]].copy()
        part["hvac_energy_kWh"] = hvac_vals
        result_parts.append(part)

    if result_parts:
        result = pd.concat(result_parts, ignore_index=True)
    else:
        result = pd.DataFrame(
            columns=["trip_id", "date", "trip_is_within_gtfs_scope", "hvac_energy_kWh"]
        )
    result["scenario"] = "TMY"
    return result[
        ["trip_id", "date", "scenario", "hvac_energy_kWh", "trip_is_within_gtfs_scope"]
    ]
