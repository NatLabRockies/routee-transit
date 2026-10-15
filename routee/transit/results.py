"""Service dates and daily block totals shared by prediction and HVAC."""

import logging
from collections.abc import Mapping
from typing import cast

import pandas as pd
from gtfsblocks import Feed

from routee.transit.vehicle_specs import VehicleSpec

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


def add_block_feasibility(
    blocks: pd.DataFrame, vehicle_specs: Mapping[str, VehicleSpec]
) -> pd.DataFrame:
    """Flag whether each block fits within the vehicle's usable energy storage.

    Parameters
    ----------
    blocks : pd.DataFrame
            Block totals from :func:`aggregate_block_predictions`.
    vehicle_specs : Mapping of str to VehicleSpec
            Capacity metadata keyed by vehicle model name. Vehicles absent from
            the mapping get a null capacity and null feasibility flags.

    Returns
    -------
    pd.DataFrame
            Input columns plus the two feasibility flags.

    Raises
    ------
    ValueError
            If a vehicle's capacity is recorded in a different unit than the one
            its energy predictions are reported in.
    """
    blocks = blocks.copy()
    usable: list[float] = []
    for vehicle, unit in zip(blocks["vehicle"], blocks["energy_unit"]):
        spec = vehicle_specs.get(vehicle)
        if spec is None:
            usable.append(float("nan"))
            continue
        if spec.capacity_unit != unit:
            raise ValueError(
                f"Vehicle '{vehicle}' reports energy in '{unit}' but its capacity "
                f"is recorded in '{spec.capacity_unit}'"
            )
        usable.append(spec.usable_capacity)

    capacity = pd.Series(usable, index=blocks.index, dtype="float64")
    known = capacity.notna()
    for suffix in ("without_deadhead", "with_deadhead"):
        feasible = blocks[f"energy_used_{suffix}"] <= capacity
        blocks[f"feasible_{suffix}"] = feasible.astype("boolean").where(known)
    return blocks


def _grouped_id_lists(
    frame: pd.DataFrame, keys: list[str], column: str, index: pd.MultiIndex
) -> pd.Series:
    """Collect each group's sorted, deduplicated non-null values from ``column``.

    Parameters
    ----------
    frame : pd.DataFrame
            Rows to aggregate, containing ``keys`` and optionally ``column``.
    keys : list of str
            Grouping columns.
    column : str
            Column whose values are collected. Missing columns yield empty lists.
    index : pd.MultiIndex
            Groups to report, in output order.

    Returns
    -------
    pd.Series
            Lists aligned to ``index``; groups with no values get an empty list.
    """
    if column not in frame:
        return pd.Series([[] for _ in range(len(index))], index=index, dtype=object)
    # Deduplicating first keeps the per-group Python list build small.
    pairs = frame[keys + [column]].dropna(subset=[column]).drop_duplicates()
    lists = pairs.sort_values(column).groupby(keys, sort=False)[column].agg(list)
    aligned = lists.reindex(index)
    return aligned.where(aligned.notna(), pd.Series([[]] * len(index), index=index))


def aggregate_block_predictions(
    trips: pd.DataFrame, vehicle_specs: Mapping[str, VehicleSpec] | None = None
) -> pd.DataFrame:
    """Sum daily revenue and deadhead demand for assigned GTFS blocks.

    Parameters
    ----------
    trips : pd.DataFrame
            Daily predictions with vehicle, date, energy and trip metadata.
    vehicle_specs : Mapping of str to VehicleSpec or None
            Capacity metadata used to flag block feasibility. If None, the
            feasibility columns are present but null.

    Returns
    -------
    pd.DataFrame
            One row per vehicle/date/block with both energy totals, feasibility
            flags, and the revenue route and agency IDs the block covers.
            Unassigned trips and blocks with no surviving revenue service are
            excluded; an empty result retains its column schema.
    """
    columns = [
        "vehicle",
        "date",
        "block_id",
        "energy_used_without_deadhead",
        "energy_used_with_deadhead",
        "energy_unit",
        "route_ids",
        "agency_ids",
    ]
    keys = ["vehicle", "date", "block_id"]
    empty = pd.DataFrame(columns=columns)
    assigned = trips.loc[usable_block_ids(trips)]
    if assigned.empty:
        return add_block_feasibility(empty, {})

    grouped = assigned.groupby(keys, sort=True)
    units = grouped["energy_unit"].agg(
        distinct="nunique", present="count", total="size"
    )
    if (units["distinct"] != 1).any() or (units["present"] != units["total"]).any():
        raise ValueError("Block predictions have mixed or missing energy units")

    totals = grouped.agg(
        energy_used_with_deadhead=("energy_used", "sum"),
        energy_unit=("energy_unit", "first"),
    )
    service = assigned.loc[assigned["trip_type"].eq("service")]
    revenue = service.groupby(keys, sort=True)["energy_used"].sum()

    # Deadhead without revenue service means the block's service trips were
    # dropped upstream, so the remaining total is not a real block.
    dropped = totals.index.difference(revenue.index)
    if not dropped.empty:
        deadhead_only_blocks = set(dropped.get_level_values("block_id"))
        sample = sorted(str(block) for block in deadhead_only_blocks)[:5]
        logger.warning(
            "Dropped %d block(s) containing only deadhead trips; their revenue "
            "service was excluded upstream (e.g. %s)",
            len(deadhead_only_blocks),
            ", ".join(sample),
        )
        totals = totals.drop(index=dropped)
    if totals.empty:
        return add_block_feasibility(empty, vehicle_specs or {})

    groups = cast(pd.MultiIndex, totals.index)
    blocks = totals.assign(
        energy_used_without_deadhead=revenue,
        route_ids=_grouped_id_lists(service, keys, "route_id", groups),
        agency_ids=_grouped_id_lists(service, keys, "agency_id", groups),
    ).reset_index()
    return add_block_feasibility(blocks[columns], vehicle_specs or {})
