import argparse
import io
import os
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from gtfsblocks import Feed

from scripts.feeds.extract_static_gtfs import GtfsExtractor

GTFS_ROUTE_TYPE_BUS = 3


def collect_active_feeds(
    extractor: GtfsExtractor,
    states: list[str],
    feed_ids: list[str],
) -> list[dict[str, Any]]:
    """Collect active US GTFS feeds from the Mobility Database.

    Parameters
    ----------
    extractor:
        Authenticated GtfsExtractor instance.
    states:
        Full state names to filter by (e.g. ``["Colorado", "Utah"]``).
        Pass an empty list to collect feeds from all US states.
    feed_ids:
        Specific MDB feed IDs to include (e.g. ``["mdb-123", "mdb-456"]``).
        Applied as a final filter after the state/all query. Pass an empty
        list to skip this filter.

    Returns
    -------
    list[dict[str, Any]]
        Raw feed records with ``status == "active"``.
    """
    base_query = "?&status=active&country_code=US"

    # If specific feed IDs are requested and no state filter is applied,
    # fetch each feed directly to avoid a slow full-catalogue query.
    if feed_ids and not states:
        active_feeds: list[dict[str, Any]] = []
        for fid in feed_ids:
            try:
                feed = extractor.query_mdb_feed(fid)
                if feed.get("status") == "active":
                    active_feeds.append(feed)
                else:
                    print(f"Feed '{fid}' exists but is not active. Skipping.")
            except Exception as exc:
                print(f"Could not fetch feed '{fid}': {exc}. Skipping.")
        return active_feeds

    if states:
        active_feeds = []
        for state in states:
            response = extractor.query_mdb_feeds(
                query=base_query + f"&subdivision_name={state}"
            )
            active_feeds.extend([r for r in response if r["status"] == "active"])
    else:
        response = extractor.query_mdb_feeds(query=base_query)
        active_feeds = [r for r in response if r["status"] == "active"]

    if feed_ids:
        id_set = set(feed_ids)
        active_feeds = [f for f in active_feeds if f["id"] in id_set]
        found_ids = {f["id"] for f in active_feeds}
        for missing in id_set - found_ids:
            print(f"Feed ID '{missing}' was not found in the queried feeds. Skipping.")

    return active_feeds


def build_feeds_summary(
    active_feeds: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build a summary DataFrame from raw feed records.

    Parameters
    ----------
    active_feeds:
        Raw feed records as returned by :func:`collect_active_feeds`.

    Returns
    -------
    pd.DataFrame
        One row per feed with id, name, provider, location, and dataset info.
    """
    feed_info: list[dict[str, Any]] = []
    for f in active_feeds:
        bbox = f["bounding_box"]
        latest_data = f["latest_dataset"]

        if latest_data is None:
            print(
                f"Feed {f['id']} does not have a latest dataset identified. "
                "Skipping this feed."
            )
            continue

        if bbox is None:
            print(
                f"Feed {f['id']} is missing a service area bounding box. "
                "Skipping this feed."
            )
            continue

        try:
            # Compile the list of states covered by this feed
            states = list(set(loc["subdivision_name"] for loc in f["locations"]))
            feed_info.append(
                {
                    "id": f["id"],
                    "name": f["feed_name"],
                    "provider": f["provider"],
                    "status": f["status"],
                    "official": f["official"],
                    "latest_dataset_id": f["latest_dataset"]["id"],
                    "center_latitude": 0.5
                    * (
                        f["bounding_box"]["minimum_latitude"]
                        + f["bounding_box"]["maximum_latitude"]
                    ),
                    "center_longitude": 0.5
                    * (
                        f["bounding_box"]["minimum_longitude"]
                        + f["bounding_box"]["maximum_longitude"]
                    ),
                    "states": states,
                }
            )
        except KeyError as err:
            print(
                f"KeyError: Missing key '{err.args[0]}' in feed with id "
                f"'{f.get('id', 'unknown')}'. Skipping this feed."
            )

    return pd.DataFrame(feed_info)


def compute_agency_metrics(
    dataset: Feed,
    dataset_id: str,
    bus_trips: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Compute per-agency overview metrics from a GTFS feed for bus trips only.

    Agencies with no bus trips are excluded entirely from the result.

    Parameters
    ----------
    dataset:
        Loaded GTFS feed.
    dataset_id:
        MDB dataset ID, included in every returned record for joining back to
        ``datasets.csv``.
    bus_trips:
        DataFrame of bus trips (already filtered to ``route_type == 3``).

    Returns
    -------
    list[dict[str, Any]]
        One record per agency that operates bus service, containing:

        - ``dataset_id``
        - ``agency_id``
        - ``agency_name``
        - ``trips_per_day``
        - ``avg_trip_duration_minutes``
        - ``avg_trip_distance_miles``
    """
    if bus_trips.empty:
        return []

    routes = dataset.routes  # indexed by route_id

    # Attach agency_id to each bus trip via its route
    trip_info = bus_trips[["trip_id", "route_id", "service_id"]].copy()
    if "agency_id" in routes.columns:
        trip_info = trip_info.join(routes[["agency_id"]], on="route_id")
    else:
        trip_info["agency_id"] = None

    # For single-agency feeds the agency_id column may be entirely NaN.
    # In that case, treat all trips as belonging to the sole agency and
    # use a sentinel value so groupby works correctly.
    SINGLE_AGENCY_SENTINEL = "__single__"
    all_null = trip_info["agency_id"].isna().all()
    if all_null:
        trip_info["agency_id"] = SINGLE_AGENCY_SENTINEL

    # --- trips_per_day -------------------------------------------------------
    date_sids = dataset.get_service_ids_all_dates()

    sid_trip_counts = (
        trip_info.groupby(["service_id", "agency_id"])["trip_id"]
        .nunique()
        .reset_index(name="n_trips")
    )
    trips_by_date = date_sids.merge(sid_trip_counts, on="service_id", how="inner")
    daily_trips = (
        trips_by_date.groupby(["date", "agency_id"])["n_trips"].sum().reset_index()
    )
    trips_per_day: pd.Series = daily_trips.groupby("agency_id")["n_trips"].mean()

    # --- avg_trip_duration_minutes -------------------------------------------
    bus_trip_ids = trip_info["trip_id"].unique()
    st = dataset.stop_times[dataset.stop_times["trip_id"].isin(bus_trip_ids)]
    trip_durations: pd.Series = st.groupby("trip_id")["arrival_time"].agg(
        lambda x: (x.max() - x.min()).total_seconds() / 60
    )
    trip_info_dur = trip_info.join(trip_durations.rename("duration_min"), on="trip_id")
    avg_duration: pd.Series = trip_info_dur.groupby("agency_id")["duration_min"].mean()

    # --- avg_trip_distance_miles ---------------------------------------------
    avg_distance: pd.Series = pd.Series(dtype=float)
    shape_ids: list[str] = []
    if "shape_id" in bus_trips.columns:
        shape_ids = bus_trips["shape_id"].dropna().unique().tolist()
    if shape_ids and dataset.shapes is not None and not dataset.shapes.empty:
        dataset.summarize_shapes(shape_ids)
        if (
            dataset.shapes_summary is not None
            and "service_dist" in dataset.shapes_summary.columns
        ):
            trip_shapes = trip_info.merge(
                bus_trips[["trip_id", "shape_id"]].dropna(subset=["shape_id"]),
                on="trip_id",
                how="left",
            )
            trip_shapes = trip_shapes.join(
                dataset.shapes_summary["service_dist"], on="shape_id"
            )
            avg_distance = trip_shapes.groupby("agency_id")["service_dist"].mean()

    # --- Build per-agency records --------------------------------------------
    # Build a lookup from sentinel/real agency_id -> agency_name
    agency_df = dataset.agency.copy()
    if all_null:
        # Map sentinel to the sole agency
        sole_name = agency_df["agency_name"].iloc[0]
        sole_id = (
            agency_df["agency_id"].iloc[0] if "agency_id" in agency_df.columns else None
        )
        agency_lookup: dict[str, tuple[Any, Any]] = {
            SINGLE_AGENCY_SENTINEL: (sole_id, sole_name)
        }
    else:
        has_id_col = "agency_id" in agency_df.columns
        agency_lookup = {}
        for _, row in agency_df.iterrows():
            key = str(row["agency_id"]) if has_id_col else SINGLE_AGENCY_SENTINEL
            agency_lookup[key] = (
                row["agency_id"] if has_id_col else None,
                row["agency_name"],
            )

    records: list[dict[str, Any]] = []
    for sentinel_id in trips_per_day.index:
        tpd = trips_per_day.get(sentinel_id)
        if tpd is None:
            continue

        real_id, agency_name = agency_lookup.get(sentinel_id, (sentinel_id, None))
        adur = avg_duration.get(sentinel_id)
        adist = (
            avg_distance.get(sentinel_id) if sentinel_id in avg_distance.index else None
        )

        records.append(
            {
                "dataset_id": dataset_id,
                "agency_id": real_id,
                "agency_name": agency_name,
                "trips_per_day": round(float(tpd), 1),
                "avg_trip_duration_minutes": (
                    round(float(adur), 1)
                    if adur is not None and not pd.isna(adur)
                    else None
                ),
                "avg_trip_distance_miles": (
                    round(float(adist), 1)
                    if adist is not None and not pd.isna(adist)
                    else None
                ),
            }
        )

    return records


def process_dataset(
    extractor: GtfsExtractor,
    db_root: Path,
    dataset_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fetch metadata for one dataset and, if valid, download and inspect it.

    A dataset is downloaded only when it has shapes and no validation errors.
    Bus-trip and shape coverage checks are performed on the extracted files.

    Parameters
    ----------
    extractor:
        Authenticated GtfsExtractor instance.
    db_root:
        Root directory under which extracted GTFS files are stored.
    dataset_id:
        MDB dataset ID string.

    Returns
    -------
    tuple[dict[str, Any], list[dict[str, Any]]]
        A ``(dataset_summary, agency_metrics)`` tuple where:

        - ``dataset_summary`` is a record for ``datasets.csv``
        - ``agency_metrics`` is a list of per-agency records for ``agencies.csv``
          (empty when the dataset was not downloaded or has no bus trips)
    """
    dataset_response = extractor.query_mdb_dataset(dataset_id=dataset_id)
    val_report = dataset_response["validation_report"]

    if val_report is None:
        print(f"\tValidation report is missing for dataset {dataset_id}.")
        print(dataset_response)
        has_shapes: bool | None = None
        has_errors: bool | None = None
    else:
        has_shapes = "Shapes" in val_report["features"]
        has_errors = val_report["total_error"] > 0

    summary: dict[str, Any] = {
        "id": dataset_id,
        "has_shapes": has_shapes,
        "has_errors": has_errors,
        "service_date_range_start": dataset_response["service_date_range_start"],
        "service_date_range_end": dataset_response["service_date_range_end"],
        "hosted_url": dataset_response["hosted_url"],
    }
    agency_metrics: list[dict[str, Any]] = []

    try:
        download_zip = requests.get(summary["hosted_url"], timeout=60)
        download_zip.raise_for_status()

        extract_path = db_root / dataset_id / "gtfs"
        os.makedirs(extract_path, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(download_zip.content)) as zip_ref:
            zip_ref.extractall(extract_path)
            print(f"Dataset extracted to {extract_path}")

        # Read in the full dataset
        dataset = Feed.from_dir(extract_path)
        routes = dataset.routes
        trips = dataset.trips

        bus_route_ids = routes[
            routes["route_type"] == GTFS_ROUTE_TYPE_BUS
        ].index.tolist()
        bus_trips = trips[trips["route_id"].isin(bus_route_ids)]

        if len(bus_trips) >= 1:
            print("\tDataset includes bus trips")
            summary["includes_bus_trips"] = True
        else:
            print("\tNo bus trips in dataset")
            summary["includes_bus_trips"] = False

        if "shape_id" in bus_trips.columns:
            if bus_trips["shape_id"].isna().sum() == 0:
                print("\tAll bus trips have shapes provided")
                summary["includes_all_bus_shapes"] = True
            else:
                print("\tSome bus trips are missing shapes")
                summary["includes_all_bus_shapes"] = False
        else:
            print("\tNo shapes in dataset")
            summary["includes_all_bus_shapes"] = False

        # Add the dataset summary
        feed_overview_dict = dataset.get_feed_overview_dict()
        # Start and end dates are already covered
        del feed_overview_dict["start_date"]
        del feed_overview_dict["end_date"]
        summary.update(feed_overview_dict)

        # Add list of agencies
        summary["agency_names"] = list(dataset.agency["agency_name"].unique())

        # If there's more than one agency, only include agencies that have bus
        # service included. We treat this as a separate check because agency_id
        # is allowed to be NA if there is only one agency in the feed.
        if len(summary["agency_names"]) > 1:
            agency_ids = list(dataset.agency["agency_id"].dropna().unique())
            if agency_ids:
                # Double check these IDs are represented in bus routes
                agency_ids_bus = list(
                    dataset.routes[dataset.routes["route_type"] == GTFS_ROUTE_TYPE_BUS][
                        "agency_id"
                    ].unique()
                )
                agency_names_bus = (
                    dataset.agency.set_index("agency_id")
                    .loc[agency_ids_bus]["agency_name"]
                    .tolist()
                )
                summary["agency_names"] = agency_names_bus

        # Compute per-agency metrics (bus trips only, excludes agencies with
        # no bus service)
        if summary.get("includes_bus_trips"):
            agency_metrics = compute_agency_metrics(dataset, dataset_id, bus_trips)

    except (ValueError, FileNotFoundError):
        pass  # return summary as-is

    return summary, agency_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Gather GTFS feeds and datasets.")
    parser.add_argument(
        "--db_root",
        type=str,
        default="reports/mdb",
        help="Root directory for storing GTFS datasets and feed info.",
    )
    parser.add_argument(
        "--states",
        type=str,
        nargs="*",
        default=[],
        help=(
            "State(s) from which to pull feeds (full name). "
            "Provide multiple states separated by spaces. "
            "Leave empty for all states."
        ),
    )
    parser.add_argument(
        "--feed_ids",
        type=str,
        nargs="+",
        default=[],
        help=(
            "Specific MDB feed ID(s) to process (e.g. mdb-123 mdb-456). "
            "When provided, only these feeds are processed. "
            "Can be combined with --states to restrict the initial query."
        ),
    )
    args = parser.parse_args()

    db_root = Path(args.db_root)
    extractor = GtfsExtractor()

    active_feeds = collect_active_feeds(
        extractor=extractor,
        states=args.states,
        feed_ids=args.feed_ids,
    )
    print(f"Collected {len(active_feeds)} active feed(s).")

    if not active_feeds:
        print("Warning: No active feeds found. Exiting.")
        return

    feeds_df = build_feeds_summary(active_feeds)

    all_dataset_summaries: list[dict[str, Any]] = []
    all_agency_metrics: list[dict[str, Any]] = []
    for d_id in feeds_df["latest_dataset_id"].tolist():
        dataset_summary, agency_metrics = process_dataset(extractor, db_root, d_id)
        all_dataset_summaries.append(dataset_summary)
        all_agency_metrics.extend(agency_metrics)

    datasets_df = pd.DataFrame(all_dataset_summaries)
    agencies_df = pd.DataFrame(all_agency_metrics)

    # Write results summary tables
    os.makedirs(db_root, exist_ok=True)
    feeds_df.to_csv(db_root / "feeds.csv", index=False)
    datasets_df.to_csv(db_root / "datasets.csv", index=False)
    if not agencies_df.empty:
        agencies_df.to_csv(db_root / "agencies.csv", index=False)
    print(
        f"Results written to {db_root}/feeds.csv, {db_root}/datasets.csv, and {db_root}/agencies.csv"
    )


if __name__ == "__main__":
    main()
