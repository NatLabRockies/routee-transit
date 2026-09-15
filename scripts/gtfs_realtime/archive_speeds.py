"""Estimate link speeds from the gtfsrt.io archive using the established workflow.

This is the archive-based counterpart to :mod:`aggregate_agency_records`. Instead
of scraping a live feed to JSONL, it pulls historical vehicle-position parquet and
period-exact static GTFS (schedules) directly from the public gtfsrt.io GCS bucket
(``parquet.gtfsrt.io``), then reuses the same map-matching / speed-estimation
library in :mod:`realtime_speeds`.

Archive layout (public, no auth):
- VP:     ``vehicle_positions/date=YYYY-MM-DD/base64url=<b64(rt_url)>/data.parquet``
- Static: ``schedules/base64url=<b64(static_url)>/_feed_digest=v1:<hash>/<file>.parquet``

For each agency we download the schedule snapshot (``_feed_digest``) that was live
on the requested dates, materialise it as a standard ``static/`` GTFS directory,
download the requested VP days, and run per-trip link-speed estimation.

The work is split into two stages so the Overpass-heavy network download is done
once and the analysis can be re-run later for different dates:

- ``--stage networks``: prefetch + cache each agency's OSM network (uses a stable,
  date-independent bbox so the analysis stage reuses the cache). Run sequentially
  to stay gentle on Overpass.
- ``--stage analyze``: estimate speeds for ``--dates`` or ``--sample-weeks N``,
  reusing the cached networks (no Overpass). Output is a parquet dataset
  partitioned as ``link_observations/agency=<slug>/date=<date>.parquet`` — a
  tidy per-link-per-trip table ready for speed-model fitting.

Example::

    # Stage 1 (once): fetch all agency networks
    .pixi/envs/hpc/bin/python scripts/gtfs_realtime/archive_speeds.py \
        --stage networks --agency all

    # Stage 2 (re-runnable): 3 weeks per agency, model-ready parquet
    .pixi/envs/hpc/bin/python scripts/gtfs_realtime/archive_speeds.py \
        --stage analyze --agency all --sample-weeks 3
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd
import pyarrow.parquet as pq

# Make sibling library importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from realtime_speeds import (
    build_compass_app,
    get_link_speeds_for_trip,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
log = logging.getLogger("archive_speeds")

BUCKET = "parquet.gtfsrt.io"
_GCS_LIST = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
_GCS_OBJ = f"https://storage.googleapis.com/{BUCKET}"

# GTFS files we materialise from the schedule parquet snapshot.
_STATIC_FILES = ("trips", "shapes", "stops", "stop_times")

# Skip a day when fewer than this fraction of distinct RT trip_ids resolve to
# static trips.txt (some feeds publish RT trip_ids in a different id space).
MIN_TRIP_MATCH_RATE = 0.5


@dataclass(frozen=True)
class Agency:
    """A transit agency's archive coordinates."""

    slug: str
    name: str
    vp_url: str  # realtime vehicle-positions producer URL
    schedule_url: str  # static GTFS producer URL


# Bus agencies present in the gtfsrt.io archive with both VP and schedules.
AGENCIES: dict[str, Agency] = {
    "dayton": Agency(
        "dayton",
        "Greater Dayton RTA",
        "https://ridetime.greaterdaytonrta.org/gtfsrt/vehicles",
        "https://proc.greaterdaytonrta.org/gtfs/google_transit.zip",
    ),
    "citybus-lafayette": Agency(
        "citybus-lafayette",
        "CityBus of Greater Lafayette",
        "https://bus.gocitybus.com/GTFSRT/GTFS_VehiclePositions.pb",
        "https://bus.gocitybus.com/GTFSRT/citybus-lafayette-in-us.zip",
    ),
    "mountainline": Agency(
        "mountainline",
        "Mountain Line (Missoula)",
        "https://bt.mountainline.com/gtfsrt/vehicles",
        "http://www.mountainline.com/files/MUTD_GTFS.zip",
    ),
    "bigbluebus": Agency(
        "bigbluebus",
        "Big Blue Bus",
        "https://cleverapi.bigbluebus.com/gtfsrt/vehicles",
        "http://gtfs.bigbluebus.com/current.zip",
    ),
    "actransit": Agency(
        "actransit",
        "AC Transit",
        "https://api.actransit.org/transit/gtfsrt/vehicles",
        "https://api.actransit.org/transit/gtfs/download",
    ),
    "broward": Agency(
        "broward",
        "Broward County Transit",
        "https://bctmyride-buspas.com:8080/GTFS/VehiclePositions",
        "https://www.broward.org/bct/documents/google_transit.zip",
    ),
    "cats": Agency(
        "cats",
        "Charlotte Area Transit System",
        "https://gtfsrealtime.ridetransit.org/GTFSRealTime/Vehicle/VehiclePositions.pb",
        "https://gtfsrealtime.ridetransit.org/GTFSStatic/api/GTFSDownload/GTFS.zip",
    ),
    "gcrta": Agency(
        "gcrta",
        "Greater Cleveland RTA",
        "https://gtfs-rt.gcrta.vontascloud.com/TMGTFSRealTimeWebService/Vehicle/VehiclePositions.pb",
        "https://www.riderta.com/sites/default/files/gtfs/latest/google_transit.zip",
    ),
    "metro-transit-mpls": Agency(
        "metro-transit-mpls",
        "Metro Transit (Minneapolis)",
        "https://svc.metrotransit.org/mtgtfs/vehiclepositions.pb",
        "https://svc.metrotransit.org/mtgtfs/gtfs.zip",
    ),
    "metro-transit-madison": Agency(
        "metro-transit-madison",
        "Metro Transit (Madison)",
        "https://metromap.cityofmadison.com/gtfsrt/vehicles",
        "http://transitdata.cityofmadison.com/GTFS/mmt_gtfs.zip",
    ),
    "wrta-youngstown": Agency(
        "wrta-youngstown",
        "Western Reserve Transit Authority (Youngstown)",
        "https://myvalleystops.wrtaonline.com/infopoint/GTFS-Realtime.ashx?Type=VehiclePosition",
        "https://myvalleystops.wrtaonline.com/InfoPoint/gtfs-zip.ashx",
    ),
}


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------


def _b64url(url: str) -> str:
    """gtfsrt.io key: urlsafe base64 of the producer URL, no padding."""
    return base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")


# Overpass endpoint(s). Default is the main instance (the only one reachable from
# restricted HPC egress); set ROUTEE_OVERPASS_URLS (comma-separated) to rotate
# through additional reachable mirrors. Matches the predictor's convention.
_DEFAULT_OVERPASS_ENDPOINTS = ("https://overpass-api.de/api",)


def _overpass_endpoints() -> list[str]:
    env = os.environ.get("ROUTEE_OVERPASS_URLS", "").strip()
    if env:
        return [u.strip() for u in env.split(",") if u.strip()]
    return list(_DEFAULT_OVERPASS_ENDPOINTS)


def configure_osmnx(cache_dir: Path) -> None:
    """Enable a persistent OSM cache and respect Overpass rate limits.

    Caching means a bbox downloaded once is reused on retries/re-runs, so we hit
    Overpass as little as possible; rate limiting makes osmnx pause per the
    server's advertised status instead of hammering it.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir)
    ox.settings.overpass_rate_limit = True
    ox.settings.requests_timeout = 180
    ox.settings.overpass_url = _overpass_endpoints()[0]


def _build_compass_app_safe(
    bbox: tuple[float, float, float, float], slug: str, base_wait: float = 30.0
):
    """Build the CompassApp for *bbox*, rotating endpoints and backing off on failure.

    Uses the osmnx download cache, so once a bbox has been fetched (e.g. by the
    network-prefetch stage) later calls reuse it without touching Overpass.
    """
    endpoints = _overpass_endpoints()
    retries = max(5, len(endpoints) + 1)
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        endpoint = endpoints[(attempt - 1) % len(endpoints)]
        ox.settings.overpass_url = endpoint
        try:
            return build_compass_app(bbox=bbox)
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            transient = any(
                k in msg
                for k in (
                    "overpass",
                    "connection",
                    "refused",
                    "timed out",
                    "timeout",
                    "max retries",
                    "429",
                    "rate",
                    "temporarily",
                )
            )
            if attempt == retries or not transient:
                raise
            wait = base_wait * attempt
            log.warning(
                "[%s] OSM download failed on %s (attempt %d/%d): %s \u2014 "
                "rotating endpoint, retrying in %.0fs",
                slug,
                endpoint,
                attempt,
                retries,
                type(exc).__name__,
                wait,
            )
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def _http_get(url: str, timeout: int = 120, retries: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001 - network flakiness, retry
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url}") from last


def _gcs_list(prefix: str) -> tuple[list[str], list[str]]:
    """Return (subprefixes, object_names) directly under *prefix*."""
    prefixes: list[str] = []
    items: list[str] = []
    token: str | None = None
    while True:
        url = (
            f"{_GCS_LIST}?prefix={urllib.parse.quote(prefix)}"
            "&delimiter=/&maxResults=1000"
        )
        if token:
            url += f"&pageToken={token}"
        payload = json.loads(_http_get(url).decode())
        prefixes += payload.get("prefixes") or []
        items += [i["name"] for i in payload.get("items", [])]
        token = payload.get("nextPageToken")
        if not token:
            return prefixes, items


def _gcs_get(object_name: str) -> bytes:
    return _http_get(f"{_GCS_OBJ}/{urllib.parse.quote(object_name)}")


# ---------------------------------------------------------------------------
# Static schedule snapshot selection + materialisation
# ---------------------------------------------------------------------------


def _list_schedule_digests(schedule_url: str) -> list[tuple[str, str]]:
    """All schedule snapshot ``(date_retrieved, prefix)`` pairs, sorted by date."""
    b64 = _b64url(schedule_url)
    digest_prefixes, _ = _gcs_list(f"schedules/base64url={b64}/")
    if not digest_prefixes:
        raise RuntimeError(f"No schedule snapshots archived for {schedule_url}")
    snapshots: list[tuple[str, str]] = []
    for prefix in digest_prefixes:
        meta = json.loads(_gcs_get(prefix + "metadata.json").decode())
        snapshots.append((meta.get("date_retrieved", ""), prefix))
    snapshots.sort()
    return snapshots


def _digest_for_date(snapshots: list[tuple[str, str]], date: str) -> str:
    """Pick the snapshot prefix live on *date*.

    Chooses the digest with the latest ``date_retrieved`` <= *date*; falls back
    to the earliest snapshot when *date* predates every snapshot (trip_ids are
    usually still recoverable from the oldest available schedule).
    """
    chosen = snapshots[0][1]
    for retrieved, prefix in snapshots:
        # date_retrieved is ISO (e.g. 2026-07-19T05:11:39Z); compare date part.
        if retrieved[:10] <= date:
            chosen = prefix
        else:
            break
    return chosen


def agency_network_bbox(
    agency: Agency, out_root: Path, buffer_deg: float = 0.05, round_to: float = 0.01
) -> tuple[float, float, float, float]:
    """Stable ``(west, south, east, north)`` OSM bbox for an agency.

    Derived from the *latest* schedule snapshot's shapes and rounded to
    ``round_to`` degrees so it is date-independent and identical across runs —
    which lets the analysis stage reuse the network fetched by the prefetch
    stage (same bbox -> same osmnx cache key). Cached to ``network_bbox.json``.
    """
    cache_file = out_root / agency.slug / "network_bbox.json"
    if cache_file.exists():
        return tuple(json.loads(cache_file.read_text()))  # type: ignore[return-value]

    digest = _list_schedule_digests(agency.schedule_url)[-1][1]
    raw = _gcs_get(f"{digest}shapes.parquet")
    s = pq.read_table(
        io.BytesIO(raw), columns=["shape_pt_lat", "shape_pt_lon"]
    ).to_pandas()
    lat = pd.to_numeric(s["shape_pt_lat"], errors="coerce").dropna()
    lon = pd.to_numeric(s["shape_pt_lon"], errors="coerce").dropna()
    west = math.floor(lon.min() / round_to) * round_to - buffer_deg
    south = math.floor(lat.min() / round_to) * round_to - buffer_deg
    east = math.ceil(lon.max() / round_to) * round_to + buffer_deg
    north = math.ceil(lat.max() / round_to) * round_to + buffer_deg
    bbox = (round(west, 4), round(south, 4), round(east, 4), round(north, 4))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(list(bbox)))
    return bbox


def agency_vp_date_range(agency: Agency) -> tuple[str, str]:
    """(date_min, date_max) of an agency's archived vehicle positions."""
    inv = json.loads(_gcs_get("inventory.json").decode())
    for d in inv:
        if d.get("feed_type") == "vehicle_positions" and d.get("url") == agency.vp_url:
            return d["date_min"], d["date_max"]
    raise RuntimeError(f"{agency.slug}: no vehicle_positions entry in inventory")


def sample_weeks(date_min: str, date_max: str, n_weeks: int = 3) -> list[str]:
    """Return the dates of *n_weeks* full Mon–Sun weeks spread across a range.

    Weeks are anchored at evenly spaced fractions of the available span so
    they land at different times of year when the range is long, and cluster
    (de-duplicated) when it is short.
    """
    start, end = pd.Timestamp(date_min), pd.Timestamp(date_max)
    span = max((end - start).days - 6, 0)  # leave room for a 7-day week
    fracs = [0.5] if n_weeks == 1 else [i / (n_weeks - 1) for i in range(n_weeks)]
    out: list[str] = []
    for f in fracs:
        anchor = start + pd.Timedelta(days=round(f * span))
        monday = anchor - pd.Timedelta(days=anchor.dayofweek)
        if monday < start:
            monday += pd.Timedelta(days=7)
        for k in range(7):
            day = monday + pd.Timedelta(days=k)
            if start <= day <= end:
                out.append(day.strftime("%Y-%m-%d"))
    return sorted(set(out))


def fetch_network(agency: Agency, out_root: Path) -> None:
    """Stage 1: warm the OSM + elevation caches for an agency (date-independent)."""
    log.info("=== fetch network: %s (%s) ===", agency.name, agency.slug)
    bbox = agency_network_bbox(agency, out_root)
    _, edge_attr_df = _build_compass_app_safe(bbox, agency.slug)
    log.info(
        "[%s] network cached — %d edges, bbox=%s",
        agency.slug,
        len(edge_attr_df),
        bbox,
    )


def prepare_static(agency: Agency, digest_prefix: str, out_root: Path) -> Path:
    """Materialise one schedule snapshot into ``<out>/static/<digest>/``.

    Keyed by digest (not by requested dates) and skipped when already present,
    so re-running the analysis for new dates reuses previously fetched eras.
    """
    digest_label = (
        digest_prefix.split("_feed_digest=")[-1].rstrip("/").replace(":", "_")
    )
    static_dir = out_root / agency.slug / "static" / digest_label
    if all((static_dir / f"{name}.txt").exists() for name in _STATIC_FILES):
        return static_dir

    static_dir.mkdir(parents=True, exist_ok=True)
    log.info("[%s] static snapshot: %s", agency.slug, digest_label[:22])
    for name in _STATIC_FILES:
        raw = _gcs_get(f"{digest_prefix}{name}.parquet")
        df = pq.read_table(io.BytesIO(raw)).to_pandas()
        df.to_csv(static_dir / f"{name}.txt", index=False)
    return static_dir


def load_static(static_dir: Path):
    """Load static GTFS the same way the JSONL pipeline does."""
    trips_df = pd.read_csv(
        static_dir / "trips.txt", dtype={"trip_id": str, "shape_id": str}
    ).set_index("trip_id")
    shapes_df = pd.read_csv(static_dir / "shapes.txt", dtype={"shape_id": str})
    stop_times_df = pd.read_csv(
        static_dir / "stop_times.txt", dtype={"trip_id": str, "stop_id": str}
    )
    stops_df = pd.read_csv(static_dir / "stops.txt", dtype={"stop_id": str}).set_index(
        "stop_id"
    )
    log.info(
        "loaded static: %d trips, %d shape points, %d stops",
        len(trips_df),
        len(shapes_df),
        len(stops_df),
    )
    return trips_df, shapes_df, stop_times_df, stops_df


# ---------------------------------------------------------------------------
# Vehicle-position download + adaptation
# ---------------------------------------------------------------------------


def download_vp_day(agency: Agency, date: str, out_root: Path) -> Path | None:
    """Download one day's vehicle-position parquet; return local path (or None)."""
    vp_dir = out_root / agency.slug / "vp"
    vp_dir.mkdir(parents=True, exist_ok=True)
    dest = vp_dir / f"{date}.parquet"
    if dest.exists():
        return dest

    b64 = _b64url(agency.vp_url)
    prefix = f"vehicle_positions/date={date}/base64url={b64}/"
    _, objects = _gcs_list(prefix)
    parquet_objs = [o for o in objects if o.endswith(".parquet")]
    if not parquet_objs:
        log.warning("[%s] no VP archived for %s", agency.slug, date)
        return None

    tables = [pq.read_table(io.BytesIO(_gcs_get(o))) for o in parquet_objs]
    import pyarrow as pa

    pq.write_table(pa.concat_tables(tables, promote_options="default"), dest)
    return dest


def build_rt_df(
    vp_path: Path, trips_df: pd.DataFrame
) -> tuple[pd.DataFrame, float, int]:
    """Load a day's VP parquet into the pipeline frame and report trip_id match rate.

    Returns ``(df, match_rate, n_rt_trips)`` where *df* is filtered to trips present
    in static trips.txt and *match_rate* is the fraction of distinct RT trip_ids
    that resolved to static \u2014 used to skip feeds/days whose RT trip_ids live in a
    different id space than the static schedule.
    """
    df = pq.read_table(vp_path).to_pandas()
    df["trip_id"] = df["trip_id"].astype(str)

    rt_ids = (
        df["trip_id"]
        .replace({"": np.nan, "None": np.nan, "nan": np.nan})
        .dropna()
        .unique()
    )
    n_rt_trips = len(rt_ids)
    matched = int(np.isin(rt_ids, trips_df.index.values).sum())
    match_rate = matched / n_rt_trips if n_rt_trips else 0.0

    # Zero lat/lon are sentinels for missing fixes.
    zero = (df["latitude"] == 0) | (df["longitude"] == 0)
    df.loc[zero, ["latitude", "longitude"]] = np.nan

    # Some feeds leave route_id empty in RT; recover it from static trips.txt.
    route = df.get("route_id")
    if route is None or route.replace("", np.nan).isna().all():
        df = df.drop(columns=[c for c in ["route_id"] if c in df.columns])
        df = df.merge(
            trips_df[["route_id"]], left_on="trip_id", right_index=True, how="left"
        )

    df = df[df["trip_id"].isin(trips_df.index)]
    return df, match_rate, n_rt_trips


def _peek_trip_match_rate(vp_path: Path, trips_df: pd.DataFrame) -> tuple[float, int]:
    """Fraction of distinct RT trip_ids present in static (reads only trip_id)."""
    ids = pq.read_table(vp_path, columns=["trip_id"]).to_pandas()["trip_id"].astype(str)
    rt_ids = ids.replace({"": np.nan, "None": np.nan, "nan": np.nan}).dropna().unique()
    n = len(rt_ids)
    if not n:
        return 0.0, 0
    matched = int(np.isin(rt_ids, trips_df.index.values).sum())
    return matched / n, n


# ---------------------------------------------------------------------------
# Per-agency run
# ---------------------------------------------------------------------------


def run_agency(agency: Agency, dates: list[str], out_root: Path) -> None:
    """Full pipeline for one agency across the requested dates.

    Agencies revise their static GTFS (and often renumber trip_ids) several
    times a year, so a multi-week sample spanning months can straddle several
    schedule "eras". Each date is matched to the schedule snapshot that was
    live on it (rather than using one snapshot for the whole run), so later
    weeks aren't wrongly skipped as a trip_id mismatch against a stale schedule.
    """
    log.info("=== %s (%s) — %d day(s) ===", agency.name, agency.slug, len(dates))
    snapshots = _list_schedule_digests(agency.schedule_url)
    era_for_date = {d: _digest_for_date(snapshots, d) for d in dates}
    n_eras = len(set(era_for_date.values()))
    if n_eras > 1:
        log.info("[%s] requested dates span %d schedule era(s)", agency.slug, n_eras)

    era_static: dict[str, tuple] = {}

    def _static_for(digest: str):
        if digest not in era_static:
            static_dir = prepare_static(agency, digest, out_root)
            era_static[digest] = load_static(static_dir)
        return era_static[digest]

    # Trip_id match pre-check BEFORE any OSM/Overpass download: skip mismatched
    # days, and skip the whole feed (no OSM download) when none are usable.
    usable_dates: list[str] = []
    for date in dates:
        vp_path = download_vp_day(agency, date, out_root)
        if vp_path is None:
            continue
        trips_df = _static_for(era_for_date[date])[0]
        match_rate, n_rt_trips = _peek_trip_match_rate(vp_path, trips_df)
        if n_rt_trips == 0:
            log.warning("[%s] %s: no RT trips \u2014 skipping", agency.slug, date)
            continue
        if match_rate < MIN_TRIP_MATCH_RATE:
            log.warning(
                "[%s] %s: only %.0f%% of %d RT trip_ids match static (< %.0f%%) \u2014 "
                "skipping day (RT trip_id space likely differs from static)",
                agency.slug,
                date,
                100 * match_rate,
                n_rt_trips,
                100 * MIN_TRIP_MATCH_RATE,
            )
            continue
        usable_dates.append(date)

    if not usable_dates:
        log.warning(
            "[%s] no days with sufficient trip_id match \u2014 skipping feed "
            "(no OSM download)",
            agency.slug,
        )
        return

    bbox = agency_network_bbox(agency, out_root)
    log.info("[%s] building CompassApp (bbox=%s)…", agency.slug, bbox)
    app, edge_attr_df = _build_compass_app_safe(bbox, agency.slug)
    log.info("[%s] CompassApp ready \u2014 %d edges", agency.slug, len(edge_attr_df))

    obs_root = out_root / "link_observations" / f"agency={agency.slug}"
    obs_root.mkdir(parents=True, exist_ok=True)
    n_days = 0

    for date in usable_dates:
        trips_df, _shapes_df, stop_times_df, stops_df = _static_for(era_for_date[date])
        vp_path = download_vp_day(agency, date, out_root)
        rt_df, match_rate, _ = build_rt_df(vp_path, trips_df)
        n_trips = rt_df["trip_id"].nunique()
        log.info(
            "[%s] %s: %d trips (%.0f%% of RT matched static)",
            agency.slug,
            date,
            n_trips,
            100 * match_rate,
        )

        t0 = time.time()
        results: list[pd.DataFrame] = []
        for _route_id, route_rt in rt_df.groupby("route_id"):
            for tid in route_rt["trip_id"].unique():
                res = get_link_speeds_for_trip(
                    tid,
                    rt_df=rt_df,
                    trips_df=trips_df,
                    app=app,
                    edge_attr_df=edge_attr_df,
                    stop_times_df=stop_times_df,
                    stops_df=stops_df,
                )
                if not res.empty and "_skip_reason" not in res.columns:
                    results.append(res)

        if not results:
            log.warning("[%s] %s: no usable link speeds", agency.slug, date)
            continue

        day_df = pd.concat(results, ignore_index=True)
        # Geometry -> WKT so each row is plain columnar parquet; agency/date come
        # from the hive-partitioned path, not duplicated as columns.
        if "geom" in day_df.columns:
            day_df["geom"] = day_df["geom"].apply(
                lambda g: g.wkt if g is not None else None
            )
        out_path = obs_root / f"date={date}" / "part.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        day_df.to_parquet(out_path, index=False)
        n_days += 1
        log.info(
            "[%s] %s: %d link observations → %s (%.1fs)",
            agency.slug,
            date,
            len(day_df),
            out_path.relative_to(out_root),
            time.time() - t0,
        )

    if n_days == 0:
        log.warning("[%s] no usable data across all dates", agency.slug)
    else:
        log.info("[%s] wrote %d day(s) of link observations", agency.slug, n_days)


def _parse_dates(spec: str) -> list[str]:
    """Parse a comma-separated date list or ``START:END`` inclusive range."""
    if ":" in spec:
        start, end = spec.split(":", 1)
        days = pd.date_range(start, end, freq="D")
        return [d.strftime("%Y-%m-%d") for d in days]
    return [d.strip() for d in spec.split(",") if d.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agency",
        required=True,
        help="agency slug, comma-separated slugs, or 'all'. Options: "
        + ", ".join(AGENCIES),
    )
    parser.add_argument(
        "--stage",
        choices=["networks", "analyze", "both"],
        default="both",
        help="networks=prefetch OSM networks only (Overpass-heavy, run once); "
        "analyze=estimate speeds reusing cached networks; both (default)",
    )
    parser.add_argument(
        "--dates",
        default=None,
        help="comma-separated YYYY-MM-DD list or START:END range (analyze stage)",
    )
    parser.add_argument(
        "--sample-weeks",
        type=int,
        default=None,
        help="analyze stage: auto-pick N full weeks spread across each agency's "
        "archived date range (alternative to --dates)",
    )
    parser.add_argument(
        "--out-root",
        default="gtfsrt_archive_runs",
        help="output directory root (default: ./gtfsrt_archive_runs)",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    configure_osmnx(out_root / ".osmnx_cache")

    if args.agency == "all":
        selected = list(AGENCIES.values())
    else:
        slugs = [s.strip() for s in args.agency.split(",") if s.strip()]
        unknown = [s for s in slugs if s not in AGENCIES]
        if unknown:
            parser.error(f"unknown agency {unknown}. Options: {', '.join(AGENCIES)}")
        selected = [AGENCIES[s] for s in slugs]

    # Stage 1: prefetch networks — sequential and gentle so Overpass isn't hammered.
    if args.stage in ("networks", "both"):
        for i, agency in enumerate(selected):
            if i:
                time.sleep(5)
            try:
                fetch_network(agency, out_root)
            except Exception:
                log.exception("[%s] network fetch failed", agency.slug)

    # Stage 2: analysis — reuses cached networks (no Overpass), safe to parallelize
    # across agencies via a SLURM array.
    if args.stage in ("analyze", "both"):
        if not args.dates and not args.sample_weeks:
            parser.error("analyze stage requires --dates or --sample-weeks")
        for i, agency in enumerate(selected):
            if i:
                time.sleep(5)
            try:
                if args.sample_weeks:
                    dmin, dmax = agency_vp_date_range(agency)
                    dates = sample_weeks(dmin, dmax, args.sample_weeks)
                    log.info(
                        "[%s] sampled %d days (%d weeks) across %s..%s",
                        agency.slug,
                        len(dates),
                        args.sample_weeks,
                        dmin,
                        dmax,
                    )
                else:
                    dates = _parse_dates(args.dates)
                run_agency(agency, dates, out_root)
            except Exception:
                log.exception("[%s] run failed", agency.slug)


if __name__ == "__main__":
    main()
