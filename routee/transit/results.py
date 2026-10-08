"""Service dates and daily block totals shared by prediction and HVAC."""

import logging

import pandas as pd
from gtfsblocks import Feed

logger = logging.getLogger(__name__)


def usable_block_ids(trips: pd.DataFrame) -> pd.Series:
    """Return a mask excluding absent, null, empty and whitespace block IDs.

    Parameters
    ----------
    trips : pd.DataFrame
            GTFS trips, optionally containing block_id.

    Returns
    -------
    pd.Series
            Boolean mask aligned with the input index.
    """
    if "block_id" not in trips:
        return pd.Series(False, index=trips.index)
    return trips["block_id"].notna() & trips["block_id"].astype(str).str.strip().ne("")


def _densest_window(
    sorted_dates: list[pd.Timestamp], max_days: int
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if not sorted_dates:
        return None
    best_start, best_count, right = sorted_dates[0], 0, 0
    for left, start in enumerate(sorted_dates):
        end = start + pd.Timedelta(days=max_days - 1)
        while right < len(sorted_dates) and sorted_dates[right] <= end:
            right += 1
        if right - left > best_count:
            best_start, best_count = start, right - left
    return best_start, best_start + pd.Timedelta(days=max_days - 1)


def _select_full_year_window(
    sorted_dates: list[pd.Timestamp], max_days: int
) -> tuple[pd.Timestamp, pd.Timestamp]:
    dense = _densest_window(sorted_dates, max_days)
    assert dense is not None
    dates = [date for date in sorted_dates if dense[0] <= date <= dense[1]]
    first, last = dates[0], dates[-1]
    if first.year == last.year:
        return pd.Timestamp(first.year, 1, 1), pd.Timestamp(first.year, 12, 31)
    midpoint = first + (last - first) / 2
    start = midpoint - pd.Timedelta(days=(max_days - 1) // 2)
    return start, start + pd.Timedelta(days=max_days - 1)


def build_trip_dates(
    feed: Feed,
    trips_df: pd.DataFrame,
    service_date: pd.Timestamp | None = None,
    scale_to_year: bool = False,
    max_days: int = 365,
) -> pd.DataFrame:
    """Expand trips onto the existing calendar window and optional projection.

    Parameters
    ----------
    feed : gtfsblocks.Feed
            Feed providing actual dates and typical weekday service.
    trips_df : pd.DataFrame
            Trip and service identifiers, including inferred deadhead.
    service_date : pd.Timestamp or None
            Restrict to one service day instead of expanding the calendar.
    scale_to_year : bool
            Project typical service onto uncovered dates in the year window.
    max_days : int
            Width of the densest service window, default 365.

    Returns
    -------
    pd.DataFrame
            Unique trip_id/date pairs with trip_is_within_gtfs_scope flags.
    """
    missing = {"trip_id", "service_id"}.difference(trips_df.columns)
    if missing:
        raise ValueError(f"trips_df is missing required columns: {sorted(missing)}")
    calendar = feed.get_service_ids_all_dates()[["date", "service_id"]].copy()
    calendar["date"] = pd.to_datetime(calendar["date"])
    unique_dates = calendar["date"].dropna().drop_duplicates().sort_values().tolist()
    if service_date is not None:
        calendar = calendar.loc[calendar["date"].eq(service_date)].drop_duplicates()
    else:
        window = (
            _select_full_year_window(unique_dates, max_days)
            if scale_to_year and unique_dates
            else _densest_window(unique_dates, max_days)
        )
        if window is None:
            calendar = calendar.iloc[:0]
        else:
            calendar = calendar.loc[calendar["date"].between(*window)].drop_duplicates()
    calendar["trip_is_within_gtfs_scope"] = True
    if service_date is None and scale_to_year and unique_dates:
        start, end = _select_full_year_window(unique_dates, max_days)
        typical = feed.get_typical_service_by_weekday()
        weekday_services = {
            str(weekday): list(row["service_id"]) for weekday, row in typical.iterrows()
        }
        covered = set(calendar["date"])
        projected = [
            (date, service_id, False)
            for date in pd.date_range(start, end)
            if date not in covered
            for service_id in weekday_services.get(date.day_name().lower(), [])
        ]
        if projected:
            calendar = pd.concat(
                [
                    calendar,
                    pd.DataFrame(
                        projected,
                        columns=["date", "service_id", "trip_is_within_gtfs_scope"],
                    ),
                ],
                ignore_index=True,
            )
    result = calendar.merge(
        trips_df[["trip_id", "service_id"]].drop_duplicates(), on="service_id"
    )
    dates: pd.DataFrame = result[
        ["trip_id", "date", "trip_is_within_gtfs_scope"]
    ].drop_duplicates()
    return dates


def aggregate_block_predictions(trips: pd.DataFrame) -> pd.DataFrame:
    """Sum daily revenue and deadhead demand for assigned GTFS blocks.

    Parameters
    ----------
    trips : pd.DataFrame
            Daily predictions with vehicle, date, energy and trip metadata.

    Returns
    -------
    pd.DataFrame
            One row per vehicle/date/block with both energy totals, revenue route
            IDs, agency metadata and the combined calendar scope flag. Unassigned
            trips are excluded; an empty result retains its column schema.
    """
    columns = [
        "vehicle",
        "date",
        "block_id",
        "energy_used_without_deadhead",
        "energy_used_with_deadhead",
        "energy_unit",
        "routes_covered",
        "agency_id",
        "trip_is_within_gtfs_scope",
    ]
    rows: list[dict[str, object]] = []
    assigned = trips.loc[usable_block_ids(trips)]
    if assigned.empty:
        return pd.DataFrame(columns=columns)
    for _, group in assigned.groupby(["vehicle", "date", "block_id"], sort=True):
        units = group["energy_unit"].dropna().unique()
        if len(units) != 1 or group["energy_unit"].isna().any():
            raise ValueError("Block predictions have mixed or missing energy units")
        service = group.loc[group["trip_type"].eq("service")]
        agencies: list[object] = (
            service["agency_id"].dropna().drop_duplicates().tolist()
            if "agency_id" in service
            else []
        )
        if len(agencies) > 1:
            logger.warning(
                "Block %s contains multiple agencies", group.iloc[0]["block_id"]
            )
        row = group.iloc[0]
        rows.append(
            {
                "vehicle": row["vehicle"],
                "date": row["date"],
                "block_id": row["block_id"],
                "energy_used_without_deadhead": service["energy_used"].sum(),
                "energy_used_with_deadhead": group["energy_used"].sum(),
                "energy_unit": units[0],
                "routes_covered": sorted(service["route_id"].dropna().unique().tolist())
                if "route_id" in service
                else [],
                "agency_id": agencies[0] if len(agencies) == 1 else None,
                "trip_is_within_gtfs_scope": bool(
                    group["trip_is_within_gtfs_scope"].all()
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)
