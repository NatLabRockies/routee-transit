"""Export inferred deadhead trips as TODS v2.1 supplement files.

TODS (Transit Operational Data Standard) extends GTFS with supplement files
that describe non-revenue operations such as deadhead trips.  This module
writes the five supplement files needed to represent routee-transit's
inferred deadhead trips in a TODS-compatible dataset:

- ``trips_supplement.txt``       — one row per deadhead trip with TODS_trip_type
- ``routes_supplement.txt``      — route records for deadhead routes
- ``stops_supplement.txt``       — depot stops (TODS_location_type = "garage")
- ``stop_times_supplement.txt``  — origin/destination stop times for each trip
- ``shapes_supplement.txt``      — point sequences for each deadhead shape

Revenue stop endpoints are already present in the base GTFS feed and are
referenced by their existing stop_id values; they are NOT written to
``stops_supplement.txt``.
"""

from pathlib import Path

import pandas as pd

from routee.transit.gtfs_processing import timedelta_to_gtfs_time

# Map routee-transit trip_type values to TODS_trip_type field values.
# TODS uses "pull-back" (not "pull-in") for return-to-depot trips.
_TRIP_TYPE_MAP: dict[str, str] = {
    "pull-out": "pull-out",
    "pull-in": "pull-back",
    "mid_block_deadhead": "deadhead",
}
def write_tods_deadhead(
    deadhead_trips: pd.DataFrame,
    deadhead_stop_times: pd.DataFrame,
    deadhead_stops: pd.DataFrame,
    shapes: pd.DataFrame,
    gtfs_stops: pd.DataFrame,
    output_dir: Path,
    fta_depots: pd.DataFrame | None = None,
) -> None:
    """Write inferred deadhead trips as TODS v2.1 supplement files.

    Parameters
    ----------
    deadhead_trips : pd.DataFrame
        Deadhead trip records with columns ``trip_id``, ``route_id``,
        ``service_id``, ``block_id``, ``shape_id``, and ``trip_type``.
    deadhead_stop_times : pd.DataFrame
        Stop-time records for all deadhead trips.  Times may be stored as
        ``pd.Timedelta``; they are converted to HH:MM:SS strings on write.
    deadhead_stops : pd.DataFrame
        Stop records used by deadhead trips, including depot stops with
        columns ``stop_id``, ``stop_lat``, and ``stop_lon``.  Revenue stop
        endpoints are NOT expected here (they exist in the base GTFS feed).
    shapes : pd.DataFrame
        Full shapes DataFrame from the predictor (GTFS format).  Filtered
        internally to only the shape_ids referenced by deadhead trips.
    gtfs_stops : pd.DataFrame
        Stops from the base GTFS feed.  Any stop_id already present here is
        excluded from ``stops_supplement.txt`` to avoid duplication.
    output_dir : Path
        Directory to write TODS files into.  Created if it does not exist.
    fta_depots : pd.DataFrame | None, optional
        Bundled NTD facility inventory GeoDataFrame used to derive depot
        stop_ids. Its index should match the integer identifier embedded in
        those depot stop_ids. When provided, a ``depot_metadata.csv`` file is
        written containing one row per referenced NTD facility that appears in
        ``stops_supplement.txt``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- trips_supplement.txt ---
    trips = deadhead_trips.copy()
    trips["TODS_trip_type"] = trips["trip_type"].map(_TRIP_TYPE_MAP)
    trips_cols = ["route_id", "service_id", "trip_id", "block_id", "TODS_trip_type"]
    trips[trips_cols].to_csv(output_dir / "trips_supplement.txt", index=False)

    # --- routes_supplement.txt ---
    routes = trips[["route_id"]].drop_duplicates().copy()
    routes["route_long_name"] = routes["route_id"]
    routes["route_type"] = 3  # bus
    routes.to_csv(output_dir / "routes_supplement.txt", index=False)

    # --- stops_supplement.txt ---
    # Only include stops that are NOT already in the base GTFS feed.
    # After the Phase 2 fix, this is exclusively the depot stops created during
    # routing (identified by the "depot_" prefix).
    existing_stop_ids = set(gtfs_stops["stop_id"].tolist())
    new_stops = deadhead_stops[
        ~deadhead_stops["stop_id"].isin(existing_stop_ids)
    ].copy()
    # Preserve any stop_name supplied upstream (typically the NTD
    # ``Facility Name`` for depot stops); fall back to an empty string for
    # rows that lack it so the CSV column is always present.
    if "stop_name" in new_stops.columns:
        new_stops["stop_name"] = new_stops["stop_name"].fillna("")
    else:
        new_stops["stop_name"] = ""
    new_stops["location_type"] = 0  # stop/platform
    new_stops["TODS_location_type"] = new_stops["stop_id"].apply(
        lambda sid: "garage" if str(sid).startswith("depot_") else ""
    )
    stops_cols = [
        "stop_id",
        "stop_name",
        "stop_lat",
        "stop_lon",
        "location_type",
        "TODS_location_type",
    ]
    new_stops[stops_cols].to_csv(output_dir / "stops_supplement.txt", index=False)

    # --- stop_times_supplement.txt ---
    stop_times = deadhead_stop_times.copy()
    for col in ("arrival_time", "departure_time"):
        if col in stop_times.columns:
            stop_times[col] = stop_times[col].apply(timedelta_to_gtfs_time)
    st_cols = ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
    stop_times[st_cols].to_csv(output_dir / "stop_times_supplement.txt", index=False)

    # --- shapes_supplement.txt ---
    deadhead_shape_ids = set(deadhead_trips["shape_id"].dropna().tolist())
    deadhead_shapes = shapes[shapes["shape_id"].isin(deadhead_shape_ids)].copy()
    shapes_cols = [
        "shape_id",
        "shape_pt_lat",
        "shape_pt_lon",
        "shape_pt_sequence",
    ]
    present_cols = [c for c in shapes_cols if c in deadhead_shapes.columns]
    deadhead_shapes[present_cols].to_csv(
        output_dir / "shapes_supplement.txt", index=False
    )

    # --- depot_metadata.csv ---
    # Rows from the FTA Transit_Depot shapefile for each depot referenced in
    # stops_supplement.txt, keyed by the row index embedded in the stop_id.
    if fta_depots is not None and not new_stops.empty:
        depot_stop_ids = new_stops.loc[
            new_stops["stop_id"].astype(str).str.startswith("depot_"), "stop_id"
        ]
        depot_indices = (
            depot_stop_ids.astype(str)
            .str.removeprefix("depot_")
            .astype(int)
            .unique()
            .tolist()
        )
        valid_indices = [i for i in depot_indices if i in fta_depots.index]
        if valid_indices:
            depot_meta = fta_depots.loc[valid_indices].copy()
            # Add the stop_id column so the CSV is self-contained
            depot_meta["stop_id"] = "depot_" + depot_meta.index.astype(str)
            # Drop geometry column if present (not useful in a plain CSV)
            if "geometry" in depot_meta.columns:
                depot_meta = depot_meta.drop(columns=["geometry"])
            depot_meta.to_csv(output_dir / "depot_metadata.csv", index=False)
