import logging
from pathlib import Path
from typing import cast

import boto3
import geopandas as gpd
import numpy as np
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from gtfsblocks import Feed

logger = logging.getLogger(__name__)


def fetch_counties_gdf() -> gpd.GeoDataFrame:
    gdf_county = gpd.read_file(
        "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_20m.zip"
    )
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
        energy = np.trapezoid(ps, ts)  # integrate kW over hours → kWh
        energies.append(energy)

    return np.array(energies)


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


def _densest_window(
    sorted_dates: list[pd.Timestamp], max_days: int
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """
    Return the (start, end) of the ``max_days``-wide window containing the most
    service dates.  Robust to feeds with sparse outliers far in the past or future.
    Returns None if ``sorted_dates`` is empty.
    """
    if not sorted_dates:
        return None
    best_start = sorted_dates[0]
    best_count = 0
    j = 0
    for i, start in enumerate(sorted_dates):
        window_end = start + pd.Timedelta(days=max_days - 1)
        while j < len(sorted_dates) and sorted_dates[j] <= window_end:
            j += 1
        count = j - i
        if count > best_count:
            best_count = count
            best_start = start
    return best_start, best_start + pd.Timedelta(days=max_days - 1)


def _select_full_year_window(
    sorted_dates: list[pd.Timestamp], max_days: int
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Pick a full-year window for scale_to_year mode.

    Snaps to a calendar year (Jan 1 - Dec 31) when the dense block of service
    dates falls within a single year; otherwise centers a ``max_days``-wide
    window on the midpoint of the dense block.
    """
    dense = _densest_window(sorted_dates, max_days)
    assert dense is not None  # caller guarantees non-empty
    dense_start, dense_end = dense
    dense_dates = [d for d in sorted_dates if dense_start <= d <= dense_end]
    first, last = dense_dates[0], dense_dates[-1]
    if first.year == last.year:
        year = first.year
        return pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(
            year=year, month=12, day=31
        )
    midpoint = first + (last - first) / 2
    half = pd.Timedelta(days=(max_days - 1) // 2)
    return midpoint - half, midpoint - half + pd.Timedelta(days=max_days - 1)


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
        subdirectory). If None, defaults to ``~/cache/routee-transit/TMY``.
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
    if output_dir is not None:
        tmy_dir = output_dir / "TMY"
    else:
        tmy_dir = Path.home() / "cache" / "routee-transit" / "TMY"

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
    power_lookup: dict[tuple[int, int], np.ndarray] = {}
    for key, grp in avg_power.groupby(["month", "day_of_month"]):
        month_key, day_key = cast(tuple[int, int], key)
        power_lookup[(month_key, day_key)] = grp.sort_values("hour")["Power"].to_numpy()

    # Get all (date, service_id) pairs covered by the feed, capped to max_days.
    # If a specific service_date was supplied (single-date run), restrict to that
    # date only — do not re-expand to every day the service_id runs.
    all_dates_df = feed.get_service_ids_all_dates()
    if service_date is not None:
        date_service_df = all_dates_df[all_dates_df["date"] == service_date][
            ["date", "service_id"]
        ].drop_duplicates()
        date_service_df["trip_is_within_gtfs_scope"] = True
    else:
        all_unique = sorted(all_dates_df["date"].unique())
        if scale_to_year and all_unique:
            window_start, window_end = _select_full_year_window(all_unique, max_days)
        else:
            dense = _densest_window(all_unique, max_days)
            if dense is None:
                window_start = window_end = pd.Timestamp("1970-01-01")
            else:
                window_start, window_end = dense

        in_window_mask = (all_dates_df["date"] >= window_start) & (
            all_dates_df["date"] <= window_end
        )
        date_service_df = all_dates_df.loc[
            in_window_mask, ["date", "service_id"]
        ].drop_duplicates()
        date_service_df["trip_is_within_gtfs_scope"] = True

        if scale_to_year and all_unique:
            # Project the feed's typical weekday service patterns onto uncovered
            # dates within the full-year window.  Weekdays absent from the
            # typical-service map (e.g. Tuesday in an M/W/F-only feed) get no
            # projected rows, preserving the feed's real day-of-week service shape.
            typical = feed.get_typical_service_by_weekday()
            weekday_to_sids: dict[str, list[str]] = {
                weekday: list(row["service_id"]) for weekday, row in typical.iterrows()
            }
            covered_dates: set[pd.Timestamp] = set(date_service_df["date"].unique())
            all_window_dates = pd.date_range(window_start, window_end, freq="D")
            uncovered_dates = [d for d in all_window_dates if d not in covered_dates]
            projected_rows = []
            for d in uncovered_dates:
                weekday_name = d.day_name().lower()
                for sid in weekday_to_sids.get(weekday_name, []):
                    projected_rows.append((d, sid, False))
            if projected_rows:
                projected_df = pd.DataFrame(
                    projected_rows,
                    columns=["date", "service_id", "trip_is_within_gtfs_scope"],
                )
                date_service_df = pd.concat(
                    [date_service_df, projected_df], ignore_index=True
                )

    # Join dates to trips via service_id
    trips_slim = trips_df[["trip_id", "service_id"]].drop_duplicates()
    date_trips = date_service_df.merge(trips_slim, on="service_id")

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
