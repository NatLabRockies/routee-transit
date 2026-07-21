"""NTD (National Transit Database) data loading and agency matching utilities.

This module provides functions to:

- Load and filter the NTD facility inventory to bus depot locations.
- Fuzzy-match a GTFS agency name and geographic centroid to the NTD agency
  table, combining name similarity (rapidfuzz + IDF-weighted token coverage)
  with geographic proximity.
"""

import re
from pathlib import Path
from typing import TypedDict

import geopandas as gpd
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from rapidfuzz.fuzz import WRatio, ratio

# ---------------------------------------------------------------------------
# NTD facility inventory constants
# ---------------------------------------------------------------------------

# Facility types that represent bus depots, in priority order (highest first).
# Used to pre-filter and rank NTD facility records before distance matching.
_DEPOT_FACILITY_TYPES: list[str] = [
    "General Purpose Maintenance Facility/Depot",
    "Other, Administrative & Maintenance",
    "Combined Administrative and Maintenance Facility (describe in Notes)",
    "Maintenance Facility (Service and Inspection)",
]

# NTD Primary Mode codes considered "bus-operating".
# DR (demand response) and VP (vanpool) are included because many small bus
# agencies report exclusively under these modes.
_BUS_MODES: frozenset[str] = frozenset({"MB", "RB", "CB", "TB", "PB", "DR", "VP"})

# ---------------------------------------------------------------------------
# NTD agency table constants
# ---------------------------------------------------------------------------

# NTD agency table columns used for matching
_NTD_ID_COL = "NTD_ID"
_OFFICIAL_NAME_COL = "Agency_Name"
_COMMON_NAME_COL = "Common_Name"
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Matching thresholds
_NAME_SCORE_THRESHOLD = 20  # minimum rapidfuzz score (0-100) to consider a candidate
_MAX_DISTANCE_KM = (
    200  # maximum allowed distance between agency centroid and query location
)
_WRATIO_WEIGHT = 0.5
_IDF_WEIGHT = 0.0
_PROXIMITY_SCALE_KM = 75.0  # exponential decay scale for distance scoring

# Minimum IDF-weighted query-token coverage (0-100) that the winning candidate
# must share with the matched NTD name. This rejects false positives for
# agencies that are not in the NTD at all (e.g. private intercity operators
# like Megabus/FlixBus or campus shuttles), which typically share no
# distinctive name tokens with any NTD record and would otherwise be matched
# to a geographically nearby agency purely on proximity. The lowest observed
# coverage among known-correct matches is ~22 (an agency whose GTFS name shares
# only its generic "transit" token with a differently-named NTD record), while
# non-NTD false positives typically sit at 0-18, so 20 separates them cleanly.
_MIN_IDF_COVERAGE = 20.0

# Full-string similarity (rapidfuzz ``ratio``, 0-100) above which a match is
# accepted even when it shares few whole tokens with the NTD name. This
# preserves legitimate typo/spacing variants (e.g. "Soun Transt" ->
# "Sound Transit", ratio ~76) while rejecting non-NTD names that only look
# similar under partial/token matching (e.g. "Beaumont Transit" vs "Riverside
# Transit" scores WRatio 85 but full-string ratio only ~39). Unlike WRatio,
# full-string ratio does not reward matching a single shared generic token.
_STRONG_NAME_THRESHOLD = 70.0


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_FACILITIES_FILENAME = "2024 Facility Inventory_260428_1.xlsx"
_AGENCIES_FILENAME = "NTAD_National_Transit_Map_Agencies.csv"


def _ntd_facilities_path() -> Path:
    from routee.transit import ntd_path

    return ntd_path() / _FACILITIES_FILENAME


def _ntd_agencies_path() -> Path:
    from routee.transit import ntd_path

    return ntd_path() / _AGENCIES_FILENAME


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _geocode_facilities(df: pd.DataFrame) -> pd.DataFrame:
    """Geocode rows missing lat/lon via their street address using Nominatim.

    Applies a 1.1-second rate limit between requests to comply with Nominatim's
    usage policy.  Rows whose addresses cannot be resolved are left with
    ``NaN`` coordinates and will be dropped by the caller's coordinate filter.
    """
    from geopy.exc import GeocoderServiceError, GeocoderTimedOut
    from geopy.extra.rate_limiter import RateLimiter
    from geopy.geocoders import Nominatim

    missing_mask = df["Latitude"].isna() | df["Longitude"].isna()
    if not missing_mask.any():
        return df

    geocoder = Nominatim(user_agent="routee-transit-ntd")
    geocode = RateLimiter(
        geocoder.geocode, min_delay_seconds=1.1, error_wait_seconds=5.0
    )

    df = df.copy()
    for idx in df.index[missing_mask]:
        row = df.loc[idx]
        parts: list[str] = []
        for field in ("Street Address", "City", "State"):
            val = row.get(field)
            if pd.notna(val) and str(val).strip():
                parts.append(str(val).strip())
        zip_val = row.get("ZIP Code")
        if pd.notna(zip_val):
            try:
                parts.append(str(int(zip_val)))
            except (ValueError, TypeError):
                pass
        if not parts:
            continue
        address = ", ".join(parts)
        try:
            location = geocode(address)
            if location is None and parts:
                # Street type in NTD may differ from OSM (e.g. "ROAD" vs "Dr").
                # Retry with the street type suffix stripped so "6570 PORTNER ROAD"
                # becomes "6570 PORTNER", letting the geocoder resolve the type.
                street = parts[0]
                stripped_street = " ".join(street.split()[:-1])
                if stripped_street:
                    fallback_parts = [stripped_street] + parts[1:]
                    location = geocode(", ".join(fallback_parts))
            if location is not None:
                df.at[idx, "Latitude"] = float(location.latitude)
                df.at[idx, "Longitude"] = float(location.longitude)
        except (GeocoderTimedOut, GeocoderServiceError):
            pass
    return df


def load_ntd_facilities(
    ntd_id: str | None = None,
    ntd_ids: list[str] | None = None,
    geocode_missing: bool = True,
) -> gpd.GeoDataFrame:
    """Load and filter the NTD facility inventory to bus depot locations.

    Reads the bundled NTD "Facility Inventory" xlsx, retains only rows that:

    1. Belong to a bus-operating agency (``Primary Mode Served`` in
       ``{MB, RB, CB, TB, PB, DR, VP}``).
    2. Are one of the three depot facility types (general purpose depot,
       combined admin/maintenance, or service-and-inspection facility).
    3. Have valid latitude/longitude coordinates (geocoded from the street
       address when ``geocode_missing=True`` and coordinates are absent).

    Pass ``ntd_id`` to restrict to a single agency or ``ntd_ids`` for several.
    When both are omitted all bus depot facilities across all agencies are
    returned.  Passing both is an error.

    Note
    ----
    About 64 % of bus depot facilities in the NTD inventory are missing
    lat/lon.  When ``geocode_missing=True`` (the default), the function
    attempts to resolve street addresses via Nominatim before applying the
    coordinate filter, so agencies like Transfort (NTD 80011) whose records
    have no coordinates but have valid addresses are handled automatically.

    Parameters
    ----------
    ntd_id : str | None
        Zero-padded 5-digit NTD ID (e.g. ``"00001"``).  Mutually exclusive
        with ``ntd_ids``.
    ntd_ids : list[str] | None
        List of zero-padded 5-digit NTD IDs.  Facilities for all listed
        agencies are returned combined.  Mutually exclusive with ``ntd_id``.
    geocode_missing : bool
        When ``True`` (default), facilities that pass the mode/type filters
        but lack lat/lon are geocoded via Nominatim using their street address
        before the coordinate filter is applied.  Set to ``False`` to skip
        geocoding (faster, but may return no results for agencies whose NTD
        records are missing coordinates).

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

    # Apply NTD ID filter early so geocoding only covers relevant agencies
    if ntd_ids is not None:
        normalised = [nid.zfill(5) for nid in ntd_ids]
        df = df[df["NTD ID"].isin(normalised)]

    # Geocode facilities whose addresses are present but lat/lon are missing
    if geocode_missing:
        df = _geocode_facilities(df)

    # Filter to valid coordinates
    missing_after = (df["Latitude"].isna() | df["Longitude"].isna()).sum()
    df = df[df["Latitude"].notna() & df["Longitude"].notna()]

    if df.empty:
        id_desc = f" for NTD ID(s) {ntd_ids!r}" if ntd_ids is not None else ""
        geocode_hint = (
            f"  {missing_after} candidate(s) had no coordinates and geocoding "
            "failed or was skipped (geocode_missing=False)."
            if missing_after > 0
            else ""
        )
        raise ValueError(
            f"No bus depot facilities found in NTD inventory{id_desc}.{geocode_hint}"
        )

    # Attach priority rank (lower = better)
    priority_map = {ft: i for i, ft in enumerate(_DEPOT_FACILITY_TYPES)}
    df = df.copy()
    df["depot_priority"] = df["Facility Type"].map(priority_map)

    gdf = gpd.GeoDataFrame(
        df.reset_index(drop=True),
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326",
    )
    return gdf


def _load_ntd_agencies(bus_only: bool = False) -> pd.DataFrame:
    """Load the bundled NTD agency table.

    Parameters
    ----------
    bus_only:
        When ``True``, restrict to agencies that operate at least one bus mode
        according to the NTD facility inventory.  When ``False`` (default), all
        agencies are returned — name + proximity scoring is sufficient to avoid
        rail-only mismatches in practice.
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


# ---------------------------------------------------------------------------
# Fuzzy matching internals
# ---------------------------------------------------------------------------


def _tokenize_name(name: str) -> set[str]:
    """Tokenize a name into lowercase alphanumeric tokens."""
    return set(_TOKEN_RE.findall(name.casefold()))


def _normalize_name_for_ratio(name: str) -> str:
    """Normalize a name for full-string fuzzy matching."""
    return " ".join(_TOKEN_RE.findall(name.casefold()))


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    _idf_coverage: float


def match_agency_to_ntd(
    agency_name: str,
    lat: float,
    lon: float,
    name_threshold: int = _NAME_SCORE_THRESHOLD,
    max_distance_km: float = _MAX_DISTANCE_KM,
    min_idf_coverage: float = _MIN_IDF_COVERAGE,
    agency_id: str | None = None,
) -> NTDAgencyMatch:
    """Fuzzy-match an agency name and location to a row in the NTD agency table.

     Candidates are scored by a weighted combination of name similarity and
     geographic distance. Name scoring blends:

     1. Rapidfuzz ``WRatio``
     2. IDF-weighted query-token coverage, where common tokens across NTD
         agencies (e.g. "transit", "city") have less influence than rare tokens.

     Both the official legal name (``Agency_Name``) and common name
     (``Common_Name``) are considered; the higher of the two scores is used.

     Proximity uses exponential decay (``exp(-dist / scale)``) so that very
     close matches (< 10 km) are strongly preferred over distant ones.

     If ``agency_id`` is provided and, after zero-padding to 5 digits, exactly
     matches a candidate's NTD ID, that candidate receives a large bonus to
     the combined score.

     To guard against false positives for agencies that are *not* in the NTD
     (e.g. private intercity carriers or campus shuttle systems), the winning
     candidate must also share at least ``min_idf_coverage`` of the query's
     IDF-weighted name tokens with the matched NTD name. When it does not, a
     ``ValueError`` is raised rather than returning a spurious proximity-only
     match.

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
    min_idf_coverage : float
        Minimum IDF-weighted query-token coverage (0–100) the winning
        candidate must share with the matched NTD name. Matches below this
        threshold are rejected as "not in NTD" to suppress false positives for
        non-NTD agencies. Set to ``0`` to disable this guard.
    agency_id : str | None
        Optional GTFS ``agency_id``.  When it zero-pads to a valid 5-digit NTD
        ID, the matching candidate gets a strong bonus.

    Returns
    -------
    NTDAgencyMatch
        Row from the NTD agency table for the best match.
        Includes all original columns plus ``_name_score``, ``_distance_km``,
        and ``_idf_coverage``.

    Raises
    ------
    ValueError
        If no candidate passes the name threshold, if the best candidate
        exceeds ``max_distance_km``, or if the best candidate's IDF-weighted
        name-token coverage is below ``min_idf_coverage``.
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
    candidate_idf_scores = idf_scores[mask]

    # Compute great-circle distance from query point to each candidate centroid
    query_point = (lat, lon)
    distances_km = np.array(
        [
            geodesic(query_point, (row["Lat"], row["Lon"])).km
            for _, row in candidates.iterrows()
        ]
    )

    # Combined score: name similarity (weighted 0.5) + proximity bonus (0.2).
    # Proximity uses exponential decay: nearby candidates (< 10 km) score ~1.0,
    # while distant ones decay smoothly toward 0.
    proximity = np.exp(-distances_km / _PROXIMITY_SCALE_KM)
    combined = 0.5 * (candidate_name_scores / 100.0) + 0.2 * proximity

    # Agency ID bonus: if the GTFS agency_id zero-pads to an NTD ID, boost
    # that candidate so it wins when name + proximity are even close.
    if agency_id is not None:
        try:
            padded_id = str(int(agency_id)).zfill(5)
        except (ValueError, TypeError):
            padded_id = None
        if padded_id is not None:
            id_match_mask = candidates[_NTD_ID_COL].values == padded_id
            combined[id_match_mask] += 0.3

    best_pos = int(combined.argmax())
    best_distance_km = float(distances_km[best_pos])
    best_idf_coverage = float(candidate_idf_scores[best_pos])
    best_row = candidates.iloc[best_pos]

    # Full-string similarity of the winning candidate (max over official/common
    # name). Unlike the partial/token WRatio used for ranking, this does not
    # reward a single shared generic token, so it stays low for non-NTD names.
    normalized_agency_name = _normalize_name_for_ratio(agency_name)
    best_full_ratio = max(
        ratio(
            normalized_agency_name,
            _normalize_name_for_ratio(str(best_row[_OFFICIAL_NAME_COL])),
        ),
        ratio(
            normalized_agency_name,
            _normalize_name_for_ratio(str(best_row[_COMMON_NAME_COL])),
        ),
    )

    # Confidence guard: reject matches whose name overlap is too weak. Agencies
    # that are not in the NTD (private operators, campus shuttles) typically
    # share no distinctive tokens with any NTD name and would otherwise be
    # matched to a nearby agency on proximity alone. A high full-string ratio is
    # accepted as an escape hatch for typo/spacing variants that have little
    # token overlap but near-exact character similarity.
    if (
        best_idf_coverage < min_idf_coverage
        and best_full_ratio < _STRONG_NAME_THRESHOLD
    ):
        raise ValueError(
            f"Best NTD match for '{agency_name}' is "
            f"'{best_row[_OFFICIAL_NAME_COL]}' ({best_row[_COMMON_NAME_COL]}), "
            f"but the name overlap is too weak (IDF coverage "
            f"{best_idf_coverage:.0f} < min_idf_coverage={min_idf_coverage:.0f}, "
            f"full-string ratio {best_full_ratio:.0f} < "
            f"{_STRONG_NAME_THRESHOLD:.0f}). The agency is likely not present in "
            f"the NTD (e.g. a private operator or campus system). Lower "
            f"min_idf_coverage to allow weaker matches."
        )

    if best_distance_km > max_distance_km:
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
        _idf_coverage=round(best_idf_coverage, 1),
    )
