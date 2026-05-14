import re
from pathlib import Path
from typing import TypedDict

import geopandas as gpd
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from gtfsblocks import Feed
from rapidfuzz.fuzz import WRatio
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# NTD facility inventory constants
# ---------------------------------------------------------------------------

# Facility types that represent bus depots, in priority order (highest first).
# Used to pre-filter and rank NTD facility records before distance matching.
_DEPOT_FACILITY_TYPES: list[str] = [
    "General Purpose Maintenance Facility/Depot",
    "Combined Administrative and Maintenance Facility (describe in Notes)",
    "Maintenance Facility (Service and Inspection)",
]

# NTD Primary Mode codes considered "bus-operating".
# DR (demand response) and VP (vanpool) are included because many small bus
# agencies report exclusively under these modes.
_BUS_MODES: frozenset[str] = frozenset({"MB", "RB", "CB", "TB", "PB", "DR", "VP"})


def _ntd_facilities_path() -> Path:
    from routee.transit import ntd_path

    candidates = list((ntd_path()).glob("2024 Facility Inventory*.xlsx"))
    if not candidates:
        raise FileNotFoundError(
            "NTD facility inventory xlsx not found in "
            f"{ntd_path()}. Expected a file matching '2024 Facility Inventory*.xlsx'."
        )
    return candidates[0]


def load_ntd_facilities(
    ntd_id: str | None = None,
    ntd_ids: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Load and filter the NTD facility inventory to bus depot locations.

    Reads the bundled NTD "Facility Inventory" xlsx, retains only rows that:

    1. Belong to a bus-operating agency (``Primary Mode Served`` in
       ``{MB, RB, CB, TB, PB, DR, VP}``).
    2. Are one of the three depot facility types (general purpose depot,
       combined admin/maintenance, or service-and-inspection facility).
    3. Have valid latitude/longitude coordinates.

    Pass ``ntd_id`` to restrict to a single agency or ``ntd_ids`` for several.
    When both are omitted all bus depot facilities across all agencies are
    returned.  Passing both is an error.

    Parameters
    ----------
    ntd_id : str | None
        Zero-padded 5-digit NTD ID (e.g. ``"00001"``).  Mutually exclusive
        with ``ntd_ids``.
    ntd_ids : list[str] | None
        List of zero-padded 5-digit NTD IDs.  Facilities for all listed
        agencies are returned combined.  Mutually exclusive with ``ntd_id``.

    Returns
    -------
    gpd.GeoDataFrame
        Point GeoDataFrame in EPSG:4326 with columns including ``NTD ID``,
        ``Agency Name``, ``Facility Type``, ``Facility Name``, and a
        ``depot_priority`` column (0 = highest priority depot type).
    """
    if ntd_id is not None and ntd_ids is not None:
        raise ValueError("Pass either ntd_id or ntd_ids, not both.")
    if ntd_id is not None:
        ntd_ids = [ntd_id]

    path = _ntd_facilities_path()
    df = pd.read_excel(path, dtype={"NTD ID": str})

    # Normalise NTD ID to zero-padded 5-digit string
    df["NTD ID"] = df["NTD ID"].str.zfill(5)

    # Filter to bus-operating modes
    df = df[df["Primary Mode Served"].isin(_BUS_MODES)]

    # Filter to depot facility types only
    df = df[df["Facility Type"].isin(_DEPOT_FACILITY_TYPES)]

    # Filter to valid coordinates
    df = df[df["Latitude"].notna() & df["Longitude"].notna()]

    # Attach priority rank (lower = better)
    priority_map = {ft: i for i, ft in enumerate(_DEPOT_FACILITY_TYPES)}
    df = df.copy()
    df["depot_priority"] = df["Facility Type"].map(priority_map)

    if ntd_ids is not None:
        normalised = [nid.zfill(5) for nid in ntd_ids]
        df = df[df["NTD ID"].isin(normalised)]

    if df.empty:
        id_desc = f" for NTD ID(s) {ntd_ids!r}" if ntd_ids is not None else ""
        raise ValueError(f"No bus depot facilities found in NTD inventory{id_desc}.")

    gdf = gpd.GeoDataFrame(
        df.reset_index(drop=True),
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326",
    )
    return gdf


class NTDAgencyMatch(TypedDict):
    """Row from the NTD agency table returned by :func:`match_agency_to_ntd`."""

    OBJECTID: int
    NTD_ID: str
    NTAD_NTD_ID: str
    Agency_Name: str
    Common_Name: str
    City: str
    State: str
    Lon: float
    Lat: float
    In_Latest_NTAD_Upload: str
    License: str
    x: float
    y: float
    _name_score: int
    _distance_km: float


# NTD agency table columns used for matching
_NTD_ID_COL = "NTD_ID"
_OFFICIAL_NAME_COL = "Agency_Name"
_COMMON_NAME_COL = "Common_Name"
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Matching thresholds
_NAME_SCORE_THRESHOLD = 40  # minimum rapidfuzz score (0-100) to consider a candidate
_MAX_DISTANCE_KM = (
    400  # maximum allowed distance between agency centroid and query location
)
_WRATIO_WEIGHT = 0.7
_IDF_WEIGHT = 0.3


def _ntd_agencies_path() -> Path:
    from routee.transit import ntd_path

    return ntd_path() / "NTAD_National_Transit_Map_Agencies.csv"


def _load_ntd_agencies(bus_only: bool = True) -> pd.DataFrame:
    """Load the bundled NTD agency table.

    Parameters
    ----------
    bus_only:
        When ``True`` (default), restrict to agencies that operate at least one
        bus mode according to the NTD facility inventory.  This prevents
        rail-only agencies from absorbing fuzzy name matches.
    """
    agencies = pd.read_csv(_ntd_agencies_path(), dtype={_NTD_ID_COL: str})
    if not bus_only:
        return agencies

    # Derive the set of NTD IDs that operate bus modes from the facility xlsx.
    fac_path = _ntd_facilities_path()
    fac = pd.read_excel(
        fac_path, usecols=["NTD ID", "Primary Mode Served"], dtype={"NTD ID": str}
    )
    fac["NTD ID"] = fac["NTD ID"].str.zfill(5)
    bus_ntd_ids: set[str] = set(
        fac.loc[fac["Primary Mode Served"].isin(_BUS_MODES), "NTD ID"].unique()
    )
    return agencies[agencies[_NTD_ID_COL].isin(bus_ntd_ids)].reset_index(drop=True)


def _tokenize_name(name: str) -> set[str]:
    """Tokenize a name into lowercase alphanumeric tokens."""
    return set(_TOKEN_RE.findall(name.casefold()))


def _compute_token_idf(agencies: pd.DataFrame) -> dict[str, float]:
    """Compute IDF-like token weights from official/common agency names."""
    token_document_counts: dict[str, int] = {}

    for _, row in agencies.iterrows():
        official_name = str(row.get(_OFFICIAL_NAME_COL, ""))
        common_name = str(row.get(_COMMON_NAME_COL, ""))
        document_tokens = _tokenize_name(official_name) | _tokenize_name(common_name)
        for token in document_tokens:
            token_document_counts[token] = token_document_counts.get(token, 0) + 1

    n_documents = len(agencies)
    return {
        token: float(np.log((n_documents + 1) / (count + 1)) + 1.0)
        for token, count in token_document_counts.items()
    }


def _idf_query_coverage_score(
    query_tokens: set[str], candidate_name: str, token_idf: dict[str, float]
) -> float:
    """Score candidate by weighted coverage of query tokens (0-100)."""
    if not query_tokens:
        return 0.0

    candidate_tokens = _tokenize_name(candidate_name)
    if not candidate_tokens:
        return 0.0

    denominator = sum(token_idf.get(token, 1.0) for token in query_tokens)
    if denominator == 0:
        return 0.0

    numerator = sum(
        token_idf.get(token, 1.0) for token in query_tokens & candidate_tokens
    )
    return 100.0 * numerator / denominator


def match_agency_to_ntd(
    agency_name: str,
    lat: float,
    lon: float,
    name_threshold: int = _NAME_SCORE_THRESHOLD,
    max_distance_km: float = _MAX_DISTANCE_KM,
) -> NTDAgencyMatch:
    """Fuzzy-match an agency name and location to a row in the NTD agency table.

     Candidates are scored by a weighted combination of name similarity and
     geographic distance. Name scoring blends:

     1. Rapidfuzz ``WRatio``
     2. IDF-weighted query-token coverage, where common tokens across NTD
         agencies (e.g. "transit", "city") have less influence than rare tokens.

     Both the official legal name (``Agency_Name``) and common name
     (``Common_Name``) are considered; the higher of the two scores is used.

    Parameters
    ----------
    agency_name : str
        Agency name to match (e.g. from GTFS ``agency.txt`` or Mobility Database).
    lat : float
        Approximate latitude of the agency's service area (WGS84).
    lon : float
        Approximate longitude of the agency's service area (WGS84).
    name_threshold : int
        Minimum rapidfuzz ``WRatio`` score (0–100) for a candidate to
        be considered. Candidates below this threshold are discarded before
        distance scoring.
    max_distance_km : float
        If the best candidate's centroid is farther than this from ``(lat, lon)``,
        a ``ValueError`` is raised even if the name score is high.

    Returns
    -------
    NTDAgencyMatch
        Row from the NTD agency table for the best match.
        Includes all original columns plus ``_name_score`` and
        ``_distance_km``.

    Raises
    ------
    ValueError
        If no candidate passes the name threshold, or if the best candidate
        exceeds ``max_distance_km``.
    """
    agencies = _load_ntd_agencies()

    query_tokens = _tokenize_name(agency_name)
    token_idf = _compute_token_idf(agencies)

    # Score both name columns; take the higher of the two for each row.
    # WRatio handles partial/abbreviated matches robustly.
    official_scores = np.array(
        [WRatio(agency_name, name) for name in agencies[_OFFICIAL_NAME_COL].fillna("")]
    )
    common_scores = np.array(
        [WRatio(agency_name, name) for name in agencies[_COMMON_NAME_COL].fillna("")]
    )
    wratio_scores = np.maximum(official_scores, common_scores)

    # IDF coverage downweights generic words that appear in many agencies.
    official_idf_scores = np.array(
        [
            _idf_query_coverage_score(query_tokens, name, token_idf)
            for name in agencies[_OFFICIAL_NAME_COL].fillna("")
        ]
    )
    common_idf_scores = np.array(
        [
            _idf_query_coverage_score(query_tokens, name, token_idf)
            for name in agencies[_COMMON_NAME_COL].fillna("")
        ]
    )
    idf_scores = np.maximum(official_idf_scores, common_idf_scores)

    name_scores = (_WRATIO_WEIGHT * wratio_scores) + (_IDF_WEIGHT * idf_scores)

    # Pre-filter by name threshold to avoid unnecessary distance calculations
    mask = name_scores >= name_threshold
    if not mask.any():
        best_name = agencies.loc[name_scores.argmax(), _OFFICIAL_NAME_COL]
        best_score = int(name_scores.max())
        raise ValueError(
            f"No NTD agency matched '{agency_name}' above the name threshold "
            f"of {name_threshold}. Best candidate was '{best_name}' "
            f"(score={best_score}). Try lowering name_threshold or check the "
            f"agency name spelling."
        )

    candidates = agencies[mask].copy()
    candidate_name_scores = name_scores[mask]

    # Compute great-circle distance from query point to each candidate centroid
    query_point = (lat, lon)
    distances_km = np.array(
        [
            geodesic(query_point, (row["Lat"], row["Lon"])).km
            for _, row in candidates.iterrows()
        ]
    )

    # Combined score: name similarity (weighted 0.7) + proximity bonus (0.3).
    # Proximity is normalised: 0 km → 1.0, max_distance_km → 0.0, clamped beyond that.
    proximity = np.clip(1.0 - distances_km / max_distance_km, 0.0, 1.0)
    combined = 0.7 * (candidate_name_scores / 100.0) + 0.3 * proximity

    best_pos = int(combined.argmax())
    best_distance_km = float(distances_km[best_pos])

    if best_distance_km > max_distance_km:
        best_row = candidates.iloc[best_pos]
        raise ValueError(
            f"Best NTD match for '{agency_name}' is '{best_row[_OFFICIAL_NAME_COL]}' "
            f"({best_row[_COMMON_NAME_COL]}), but its centroid is "
            f"{best_distance_km:.1f} km away — exceeds max_distance_km={max_distance_km}. "
            f"Verify the lat/lon or increase max_distance_km."
        )

    row = candidates.iloc[best_pos]
    return NTDAgencyMatch(
        OBJECTID=int(row["OBJECTID"]),
        NTD_ID=str(row["NTD_ID"]),
        NTAD_NTD_ID=str(row["NTAD_NTD_ID"]),
        Agency_Name=str(row["Agency_Name"]),
        Common_Name=str(row["Common_Name"]),
        City=str(row["City"]),
        State=str(row["State"]),
        Lon=float(row["Lon"]),
        Lat=float(row["Lat"]),
        In_Latest_NTAD_Upload=str(row["In_Latest_NTAD_Upload"]),
        License=str(row["License"]),
        x=float(row["x"]),
        y=float(row["y"]),
        _name_score=int(candidate_name_scores[best_pos]),
        _distance_km=round(best_distance_km, 2),
    )


# Default path to FTA depot shapefile, relative to repository root
def get_default_depot_path() -> Path:
    """
    Return the default path to the FTA_Depot directory in the repository.

    The default depot locations come from the National Transit Database's
    "Public Transit Facilities and Stations - 2023" dataset, which contains
    depot/facility locations for transit agencies across the United States.

    Data source: https://data.transportation.gov/stories/s/gd62-jzra

    Returns
    -------
    Path
        Path to the FTA_Depot directory containing Transit_Depot.shp
    """
    from routee.transit import depot_path

    return depot_path()


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

    block_ids = trips_df["block_id"].dropna().unique().tolist()

    # Get earliest start time for each trip and merge then in to trips DF
    trip_start_times = (
        stop_times_df.groupby("trip_id")["arrival_time"].min().reset_index()
    )
    trips_with_times = trips_df.merge(trip_start_times, on="trip_id", how="left")

    # For each block id, create two deadhead trips: one from depot to first stop,
    # and one from last stop to depot.
    depot_trips = list()

    for block_id in block_ids:
        block_trips = trips_with_times[trips_with_times["block_id"] == block_id]
        # Exclude any between-trip deadhead trips that may have been added
        if "from_trip" in block_trips.columns:
            block_trips = block_trips.loc[block_trips["from_trip"].isna()]
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
        trips_df[["trip_id", "block_id"]], on="trip_id", how="right"
    )
    blocks_trips_stops = blocks_trips_stops.merge(stops_df, on="stop_id", how="left")

    blocks_trips_stops = blocks_trips_stops.sort_values(by=["block_id", "arrival_time"])
    first_stops = blocks_trips_stops.groupby("block_id").first().reset_index()
    last_stops = blocks_trips_stops.groupby("block_id").last().reset_index()

    first_stops = first_stops[
        ["block_id", "stop_id", "arrival_time", "stop_lat", "stop_lon"]
    ]
    last_stops = last_stops[
        ["block_id", "stop_id", "arrival_time", "stop_lat", "stop_lon"]
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
                "block_id",
                "stop_id",
                "nearest_depot_idx",
                "departure_time",
                "arrival_time",
            ]
        ],
        on="block_id",
    )

    deadhead_trips_df_to_depot = deadhead_trips_df[
        deadhead_trips_df.trip_type == "pull-in"
    ].copy()
    deadhead_trips_df_to_depot = deadhead_trips_df_to_depot.merge(
        to_depot[
            [
                "block_id",
                "stop_id",
                "nearest_depot_idx",
                "departure_time",
                "arrival_time",
            ]
        ],
        on="block_id",
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
