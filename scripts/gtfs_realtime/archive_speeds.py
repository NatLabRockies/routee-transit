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

Example::

    LD_PRELOAD="$PWD/.devtools/libcompat_glibc.so" \
        .pixi/envs/dev-py312/bin/python scripts/gtfs_realtime/archive_speeds.py \
        --agency dayton --dates 2026-07-14,2026-07-15

(The LD_PRELOAD shim is only needed on hosts with glibc < 2.38; see repo notes.)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Make sibling library importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from realtime_speeds import (  # noqa: E402
    aggregate_speeds_across_trips,
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


@dataclass(frozen=True)
class Agency:
    """A transit agency's archive coordinates."""

    slug: str
    name: str
    vp_url: str  # realtime vehicle-positions producer URL
    schedule_url: str  # static GTFS producer URL


# Four bus agencies present in the gtfsrt.io archive with both VP and schedules.
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
}


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------


def _b64url(url: str) -> str:
    """gtfsrt.io key: urlsafe base64 of the producer URL, no padding."""
    return base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")


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


def _select_schedule_digest(schedule_url: str, dates: list[str]) -> str:
    """Pick the schedule snapshot prefix live on the requested dates.

    Chooses the digest with the latest ``date_retrieved`` <= the earliest
    requested date; falls back to the earliest snapshot when all snapshots
    postdate the requested window (trip_ids are stable across versions).
    """
    b64 = _b64url(schedule_url)
    digest_prefixes, _ = _gcs_list(f"schedules/base64url={b64}/")
    if not digest_prefixes:
        raise RuntimeError(f"No schedule snapshots archived for {schedule_url}")

    snapshots: list[tuple[str, str]] = []  # (date_retrieved, prefix)
    for prefix in digest_prefixes:
        meta = json.loads(_gcs_get(prefix + "metadata.json").decode())
        snapshots.append((meta.get("date_retrieved", ""), prefix))
    snapshots.sort()

    earliest_requested = min(dates)
    chosen = snapshots[0][1]
    for retrieved, prefix in snapshots:
        # date_retrieved is ISO (e.g. 2026-07-19T05:11:39Z); compare date part.
        if retrieved[:10] <= earliest_requested:
            chosen = prefix
        else:
            break
    return chosen


def prepare_static(agency: Agency, dates: list[str], out_root: Path) -> Path:
    """Download the period-correct schedule snapshot into ``<out>/static/``."""
    static_dir = out_root / agency.slug / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    digest_prefix = _select_schedule_digest(agency.schedule_url, dates)
    log.info(
        "[%s] static snapshot: %s",
        agency.slug,
        digest_prefix.split("_feed_digest=")[-1].rstrip("/")[:22],
    )
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
    stops_df = pd.read_csv(
        static_dir / "stops.txt", dtype={"stop_id": str}
    ).set_index("stop_id")
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


def build_rt_df(parquet_paths: list[Path], trips_df: pd.DataFrame) -> pd.DataFrame:
    """Load VP parquet into the frame shape expected by ``get_link_speeds_for_trip``."""
    frames = [pq.read_table(p).to_pandas() for p in parquet_paths]
    df = pd.concat(frames, ignore_index=True)

    df["trip_id"] = df["trip_id"].astype(str)

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
    return df


# ---------------------------------------------------------------------------
# Per-agency run
# ---------------------------------------------------------------------------


def run_agency(agency: Agency, dates: list[str], out_root: Path) -> None:
    """Full pipeline for one agency across the requested dates."""
    log.info("=== %s (%s) — %d day(s) ===", agency.name, agency.slug, len(dates))
    static_dir = prepare_static(agency, dates, out_root)
    trips_df, shapes_df, stop_times_df, stops_df = load_static(static_dir)

    log.info("[%s] building CompassApp from shapes bbox…", agency.slug)
    app, edge_attr_df = build_compass_app(shapes_df)
    log.info("[%s] CompassApp ready — %d edges", agency.slug, len(edge_attr_df))

    agency_dir = out_root / agency.slug
    day_frames: list[pd.DataFrame] = []

    for date in dates:
        vp_path = download_vp_day(agency, date, out_root)
        if vp_path is None:
            continue
        rt_df = build_rt_df([vp_path], trips_df)
        n_trips = rt_df["trip_id"].nunique()
        log.info("[%s] %s: %d trips", agency.slug, date, n_trips)
        if n_trips == 0:
            continue

        t0 = time.time()
        results: list[pd.DataFrame] = []
        for route_id, route_rt in rt_df.groupby("route_id"):
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
        day_df["date"] = date
        day_csv = agency_dir / f"realtime_link_speeds_{date}.csv"
        day_df.to_csv(day_csv, index=False)
        day_frames.append(day_df)
        log.info(
            "[%s] %s: %d link rows → %s (%.1fs)",
            agency.slug,
            date,
            len(day_df),
            day_csv.name,
            time.time() - t0,
        )

    if not day_frames:
        log.warning("[%s] no usable data across all dates", agency.slug)
        return

    all_df = pd.concat(day_frames, ignore_index=True)
    aggregated = aggregate_speeds_across_trips(all_df)
    if not aggregated.empty:
        agg_csv = agency_dir / "realtime_link_speeds_aggregated.csv"
        aggregated.to_csv(agg_csv, index=False)
        log.info(
            "[%s] aggregated %d links → %s",
            agency.slug,
            len(aggregated),
            agg_csv.name,
        )


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
        help="agency slug, or 'all'. Options: " + ", ".join(AGENCIES),
    )
    parser.add_argument(
        "--dates",
        required=True,
        help="comma-separated YYYY-MM-DD list, or START:END inclusive range",
    )
    parser.add_argument(
        "--out-root",
        default="gtfsrt_archive_runs",
        help="output directory root (default: ./gtfsrt_archive_runs)",
    )
    args = parser.parse_args()

    dates = _parse_dates(args.dates)
    out_root = Path(args.out_root)

    if args.agency == "all":
        selected = list(AGENCIES.values())
    elif args.agency in AGENCIES:
        selected = [AGENCIES[args.agency]]
    else:
        parser.error(f"unknown agency '{args.agency}'. Options: {', '.join(AGENCIES)}")

    for agency in selected:
        try:
            run_agency(agency, dates, out_root)
        except Exception:
            log.exception("[%s] run failed", agency.slug)


if __name__ == "__main__":
    main()
