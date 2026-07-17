import geopandas as gpd
import pandas as pd
from geopy.distance import geodesic
from gtfsblocks import Feed
from shapely.geometry import Point

from routee.transit.ntd import NTDAgencyMatch, load_ntd_facilities, match_agency_to_ntd

# Re-export for backward compatibility
__all__ = [
    "load_ntd_facilities",
    "match_agency_to_ntd",
    "NTDAgencyMatch",
    "create_depot_deadhead_trips",
    "infer_depot_trip_endpoints",
    "create_depot_deadhead_stops",
]


def create_depot_deadhead_trips(
    trips_df: pd.DataFrame, stop_times_df: pd.DataFrame
) -> pd.DataFrame:
    """Create deadhead trips from and to depots for each block.

    This function essentially creates rows for the trips.txt DataFrame.
    It does not generate shape traces for them (that is handled by other
    functions in this module).

    Parameters
    ----------
    trips_df : pd.DataFrame
        trips_df of selected date route (e.g. result from read_in_gtfs).
    stop_times_df: pd.DataFrame
        stop_times df in feed resulted from read_in_gtfs.

    Returns
    -------
    pd.DataFrame: DataFrame with created deadhead trips.
    """

    # Get earliest start time for each trip and merge then in to trips DF
    trip_start_times = (
        stop_times_df.groupby("trip_id")["arrival_time"].min().reset_index()
    )
    trips_with_times = trips_df.merge(trip_start_times, on="trip_id", how="left")

    # Exclude any between-trip deadhead trips that may have been added
    if "from_trip" in trips_with_times.columns:
        trips_with_times = trips_with_times.loc[trips_with_times["from_trip"].isna()]

    # For each (service_id, block_id) pair, create two deadhead trips: one from
    # depot to first stop, and one from last stop to depot.  Partitioning by
    # service_id (not block_id alone) ensures blocks shared across multiple
    # service calendars get a distinct pull-out/pull-in per service day, so
    # downstream date-expansion (HVAC annualisation) counts them on the correct
    # calendars.  The per-block route_id/shape_id is retained so all service days
    # of a block share one depot O-D geometry (routed once via O-D dedup).
    depot_trips = list()

    for (_service_id, block_id), block_trips in trips_with_times.groupby(
        ["service_id", "block_id"], sort=False
    ):
        # Ensure trips have been sorted in chronological order
        block_trips = block_trips.sort_values(by="arrival_time")
        first_trip = block_trips.iloc[0]
        last_trip = block_trips.iloc[-1]
        # Create trip from depot to first stop
        from_depot_trip_id = f"depot_to_{first_trip['trip_id']}"
        from_depot_route = f"from_depot_{block_id}"
        from_depot_trip = {
            "trip_id": from_depot_trip_id,
            "trip_type": "pull-out",
            "route_id": from_depot_route,
            "service_id": first_trip["service_id"],
            "block_id": block_id,
            "shape_id": from_depot_route,
            "route_short_name": from_depot_route,
            "route_type": 3,  # 3 means bus
            "route_desc": f"Deadhead from depot to {first_trip['trip_id']}",
            "agency_id": first_trip.get("agency_id", None),
        }
        depot_trips.append(from_depot_trip)
        # Create trip from last stop to depot
        to_depot_trip_id = f"{last_trip['trip_id']}_to_depot"
        to_depot_route = f"to_depot_{block_id}"
        to_depot_trip = {
            "trip_id": to_depot_trip_id,
            "trip_type": "pull-in",
            "route_id": to_depot_route,
            "service_id": last_trip["service_id"],
            "block_id": block_id,
            "shape_id": to_depot_route,
            "route_short_name": to_depot_route,
            "route_type": 3,  # 3 means bus
            "route_desc": f"Deadhead from {last_trip['trip_id']} to depot",
            "agency_id": last_trip.get("agency_id", None),
        }
        depot_trips.append(to_depot_trip)

    deadhead_trips_df = pd.DataFrame(depot_trips)
    return deadhead_trips_df


def infer_depot_trip_endpoints(
    trips_df: pd.DataFrame,
    feed: Feed,
    depots_gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Add origin/destination depot geometry for each block.

    Parameters
    ----------
    trips_df: pd.DataFrame
        trips_df of selected date and route (result from read_in_gtfs).
    feed : Feed
        GTFS feed object (e.g. result from read_in_gtfs).
    depots_gdf : gpd.GeoDataFrame
        Point GeoDataFrame of candidate depot locations in EPSG:4326.  Typically
        the result of :func:`load_ntd_facilities`.  If a ``depot_priority``
        column is present (0 = best), candidate depots are first restricted to
        the highest-priority type available before distance minimisation; if no
        higher-priority depot is reachable the next tier is tried.

    Returns
    -------
    tuple[GeoDataFrame, GeoDataFrame, GeoDataFrame]
        (first_stops_gdf, last_stops_gdf, depots_gdf). The first two contain
        stop geometry and matched depot geometry.  ``depots_gdf`` is the full
        depot GeoDataFrame (EPSG:4326) so callers can look up metadata by row
        index.
    """

    # Process trips and stops dataframes in feed to get first and last stops of each block id
    trips_df = trips_df.copy()
    stop_times_df = feed.stop_times
    stops_df = feed.stops
    blocks_trips_stops = stop_times_df.merge(
        trips_df[["trip_id", "block_id", "service_id"]], on="trip_id", how="right"
    )
    blocks_trips_stops = blocks_trips_stops.merge(stops_df, on="stop_id", how="left")

    blocks_trips_stops = blocks_trips_stops.sort_values(
        by=["service_id", "block_id", "arrival_time"]
    )
    first_stops = (
        blocks_trips_stops.groupby(["service_id", "block_id"]).first().reset_index()
    )
    last_stops = (
        blocks_trips_stops.groupby(["service_id", "block_id"]).last().reset_index()
    )

    first_stops = first_stops[
        ["service_id", "block_id", "stop_id", "arrival_time", "stop_lat", "stop_lon"]
    ]
    last_stops = last_stops[
        ["service_id", "block_id", "stop_id", "arrival_time", "stop_lat", "stop_lon"]
    ]

    first_stops["geometry"] = first_stops.apply(
        lambda row: Point(row["stop_lon"], row["stop_lat"]), axis=1
    )
    last_stops["geometry"] = last_stops.apply(
        lambda row: Point(row["stop_lon"], row["stop_lat"]), axis=1
    )
    first_stops_gdf = gpd.GeoDataFrame(
        first_stops, geometry="geometry", crs="EPSG:4326"
    )
    last_stops_gdf = gpd.GeoDataFrame(last_stops, geometry="geometry", crs="EPSG:4326")

    # Ensure depot geometries are in WGS84
    if depots_gdf.crs is None:
        depots_gdf = depots_gdf.set_crs(epsg=4326)
    else:
        depots_gdf = depots_gdf.to_crs(epsg=4326)

    has_priority = "depot_priority" in depots_gdf.columns
    priority_levels: list[int] = (
        sorted(depots_gdf["depot_priority"].dropna().unique().tolist())
        if has_priority
        else []
    )

    # Create a simple mapping from depot index to geometry for fast lookup
    depots_geom_map = depots_gdf["geometry"].to_dict()

    # Project to Web Mercator (EPSG:3857) for distance computations
    proj_crs = "EPSG:3857"
    first_proj = first_stops_gdf.to_crs(proj_crs).reset_index(drop=True)
    last_proj = last_stops_gdf.to_crs(proj_crs).reset_index(drop=True)
    depots_proj = depots_gdf.to_crs(proj_crs).copy()

    best_depot_idx: dict[object, int] = {}
    for block_id, first_row in first_proj.groupby("block_id"):
        first_geom = first_row.iloc[0].geometry
        last_geom = last_proj.loc[last_proj["block_id"] == block_id, "geometry"].values[
            0
        ]

        # Compute pull-out + pull-in distance for every depot candidate
        working = depots_proj.copy()
        working["pullout"] = working.geometry.distance(first_geom)
        working["pullin"] = working.geometry.distance(last_geom)
        working["total"] = working["pullout"] + working["pullin"]

        if has_priority:
            # Pick nearest depot from the highest-priority tier that is
            # non-empty; fall through to subsequent tiers if needed.
            best_idx: int = working["total"].idxmin()
            for level in priority_levels:
                tier = working[working["depot_priority"] == level]
                if not tier.empty:
                    best_idx = int(tier["total"].idxmin())
                    break
        else:
            best_idx = int(working["total"].idxmin())

        best_depot_idx[block_id] = best_idx

    first_stops_gdf["nearest_depot_idx"] = first_stops_gdf["block_id"].map(
        best_depot_idx
    )
    last_stops_gdf["nearest_depot_idx"] = last_stops_gdf["block_id"].map(best_depot_idx)

    first_stops_gdf["geometry_origin"] = first_stops_gdf["nearest_depot_idx"].map(
        depots_geom_map
    )
    first_stops_gdf["geometry_destination"] = first_stops_gdf.geometry
    last_stops_gdf["geometry_destination"] = last_stops_gdf["nearest_depot_idx"].map(
        depots_geom_map
    )
    last_stops_gdf["geometry_origin"] = last_stops_gdf.geometry

    # Attach NTD metadata (NTD ID, agency name, facility name/type) to both
    # stop GDFs so downstream callers and outputs can identify which depot was
    # matched without having to rejoin on nearest_depot_idx themselves.
    _ntd_meta_cols: dict[str, str] = {
        "NTD ID": "depot_ntd_id",
        "Agency Name": "depot_agency_name",
        "Facility Name": "depot_facility_name",
        "Facility Type": "depot_facility_type",
    }
    for src_col, dst_col in _ntd_meta_cols.items():
        if src_col in depots_gdf.columns:
            col_map = depots_gdf[src_col].to_dict()
            first_stops_gdf[dst_col] = first_stops_gdf["nearest_depot_idx"].map(col_map)
            last_stops_gdf[dst_col] = last_stops_gdf["nearest_depot_idx"].map(col_map)

    # Set the arrival time as departure time for deadhead trip to depot for the last_stop_gdf
    last_stops_gdf["departure_time"] = last_stops_gdf["arrival_time"]
    # Drop the arrival_time column for the last_stop_gdf
    last_stops_gdf = last_stops_gdf.drop(columns=["arrival_time"])

    # Keep only relevant columns and set stop_geometry as the active geometry
    first_stops_gdf = first_stops_gdf.drop(columns=["geometry"])
    first_stops_gdf = gpd.GeoDataFrame(
        first_stops_gdf, geometry="geometry_destination", crs="EPSG:4326"
    )

    last_stops_gdf = last_stops_gdf.drop(columns=["geometry"])
    last_stops_gdf = gpd.GeoDataFrame(
        last_stops_gdf, geometry="geometry_origin", crs="EPSG:4326"
    )

    return first_stops_gdf, last_stops_gdf, depots_gdf


def create_depot_deadhead_stops(
    first_stops_gdf: gpd.GeoDataFrame,
    last_stops_gdf: gpd.GeoDataFrame,
    deadhead_trips: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create stop_times and stops for deadhead trips from and to depots.

    Parameters
    ----------
    first_stops_gdf: gpd.GeoDataFrame
        GeoDataFrame of first stops for each block, with ``geometry_origin``
        (depot) and ``geometry_destination`` (first stop) columns.
        Result from :func:`infer_depot_trip_endpoints`.
    last_stops_gdf: gpd.GeoDataFrame
        GeoDataFrame of last stops for each block, with ``geometry_origin``
        (last stop) and ``geometry_destination`` (depot) columns.
        Result from :func:`infer_depot_trip_endpoints`.
    deadhead_trips: pd.DataFrame
        Deadhead trip records from :func:`create_depot_deadhead_trips`.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A ``(stop_times_df, stops_df)`` tuple for the depot deadhead trips.
    """

    from_depot = first_stops_gdf.copy()
    to_depot = last_stops_gdf.copy()

    # Calculate distance from depot to first stop
    from_depot["distance_m"] = from_depot.apply(
        lambda row: (
            geodesic(
                (row.geometry_origin.y, row.geometry_origin.x),
                (row.geometry_destination.y, row.geometry_destination.x),
            ).meters
        ),
        axis=1,
    )
    # Calculate distance from last stop to depot
    to_depot["distance_m"] = to_depot.apply(
        lambda row: (
            geodesic(
                (row.geometry_origin.y, row.geometry_origin.x),
                (row.geometry_destination.y, row.geometry_destination.x),
            ).meters
        ),
        axis=1,
    )
    # Assume average speed of 30 km/h (to be consistant with the number adopted in gtfs_feature_processing.py)
    # to estimate travel time
    from_depot["travel_time_sec"] = (from_depot["distance_m"] / 30000) * 3600
    to_depot["travel_time_sec"] = (to_depot["distance_m"] / 30000) * 3600
    # Calculate departure time from depot for deadhead trip to first stop
    from_depot["departure_time"] = from_depot["arrival_time"] - pd.to_timedelta(
        from_depot["travel_time_sec"], unit="s"
    )
    # Calculate arrival time at depot for deadhead trip from last stop
    to_depot["arrival_time"] = to_depot["departure_time"] + pd.to_timedelta(
        to_depot["travel_time_sec"], unit="s"
    )

    # Create stop_times df for deadhead trips
    deadhead_trips_df = deadhead_trips.copy()
    deadhead_trips_df_from_depot = deadhead_trips_df[
        deadhead_trips_df.trip_type == "pull-out"
    ].copy()
    deadhead_trips_df_from_depot = deadhead_trips_df_from_depot.merge(
        from_depot[
            [
                "service_id",
                "block_id",
                "stop_id",
                "nearest_depot_idx",
                "departure_time",
                "arrival_time",
            ]
        ],
        on=["service_id", "block_id"],
    )

    deadhead_trips_df_to_depot = deadhead_trips_df[
        deadhead_trips_df.trip_type == "pull-in"
    ].copy()
    deadhead_trips_df_to_depot = deadhead_trips_df_to_depot.merge(
        to_depot[
            [
                "service_id",
                "block_id",
                "stop_id",
                "nearest_depot_idx",
                "departure_time",
                "arrival_time",
            ]
        ],
        on=["service_id", "block_id"],
    )
    deadhead_trips_df = pd.concat(
        [deadhead_trips_df_from_depot, deadhead_trips_df_to_depot], ignore_index=True
    )
    stop_times_df = pd.DataFrame(
        columns=[
            "trip_id",
            "stop_sequence",
            "arrival_time",
            "stop_id",
            "departure_time",
            "shape_dist_traveled",
        ]
    )
    stop_times_df["trip_id"] = deadhead_trips_df["trip_id"].repeat(2).values

    stop_times_df["stop_sequence"] = [1, 2] * len(deadhead_trips_df)
    stop_times_df["arrival_time"] = [
        x
        for pair in zip(
            deadhead_trips_df["departure_time"].to_list(),
            deadhead_trips_df["arrival_time"].to_list(),
        )
        for x in pair
    ]
    # For pull-out trips: stop_sequence 1 = depot stop (new), stop_sequence 2 = first
    # revenue stop (existing GTFS stop).  For pull-in trips the order is reversed.
    # Depot stops are keyed as "depot_{nearest_depot_idx}" where nearest_depot_idx is
    # the row index in the FTA shapefile.  This means all blocks that share the same
    # physical depot get the same stop_id.
    from_depot_stop_ids = [
        x
        for pair in zip(
            (
                "depot_" + deadhead_trips_df_from_depot["nearest_depot_idx"].astype(str)
            ).tolist(),
            deadhead_trips_df_from_depot["stop_id"].tolist(),
        )
        for x in pair
    ]
    to_depot_stop_ids = [
        x
        for pair in zip(
            deadhead_trips_df_to_depot["stop_id"].tolist(),
            (
                "depot_" + deadhead_trips_df_to_depot["nearest_depot_idx"].astype(str)
            ).tolist(),
        )
        for x in pair
    ]
    stop_times_df["stop_id"] = from_depot_stop_ids + to_depot_stop_ids
    stop_times_df["departure_time"] = stop_times_df["arrival_time"]
    stop_times_df["shape_dist_traveled"] = 0.0

    # Create stops df — one row per unique physical depot (keyed by nearest_depot_idx).
    # Revenue stop endpoints are already in the GTFS feed and must not be duplicated.
    # Use depot_facility_name as stop_name when available so stops_supplement.txt
    # carries a human-readable depot identifier.
    from_depot_stop_name = (
        from_depot["depot_facility_name"]
        if "depot_facility_name" in from_depot.columns
        else pd.Series([""] * len(from_depot), index=from_depot.index)
    )
    to_depot_stop_name = (
        to_depot["depot_facility_name"]
        if "depot_facility_name" in to_depot.columns
        else pd.Series([""] * len(to_depot), index=to_depot.index)
    )
    from_depot_stops = pd.DataFrame(
        {
            "stop_id": "depot_" + from_depot["nearest_depot_idx"].astype(str),
            "stop_name": from_depot_stop_name.values,
            "stop_lat": from_depot.geometry_origin.apply(lambda p: p.y).values,
            "stop_lon": from_depot.geometry_origin.apply(lambda p: p.x).values,
        }
    )
    to_depot_stops = pd.DataFrame(
        {
            "stop_id": "depot_" + to_depot["nearest_depot_idx"].astype(str),
            "stop_name": to_depot_stop_name.values,
            "stop_lat": to_depot.geometry_destination.apply(lambda p: p.y).values,
            "stop_lon": to_depot.geometry_destination.apply(lambda p: p.x).values,
        }
    )
    stops_df = (
        pd.concat([from_depot_stops, to_depot_stops])
        .drop_duplicates(subset="stop_id")
        .reset_index(drop=True)
    )

    return stop_times_df, stops_df
