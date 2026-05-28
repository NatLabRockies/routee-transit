"""TODS Block Visualizer — Streamlit app.

Walk through every trip in a transit block — including inferred pull-out,
pull-back, and mid-block deadhead trips — and display each trip's route on
an interactive folium map.  Pull-out and pull-back trips show the selected
depot labelled with its NTD facility name and location type.

Prerequisites (not bundled with routee-transit):
    pip install streamlit streamlit-folium

Run from the repository root:
    streamlit run scripts/visualize_tods.py
"""

from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------

_TYPE_LABEL: dict[str, str] = {
    "pull-out": "Pull-out",
    # pull-in = routee-transit internal name; pull-back = TODS name — treat identically
    "pull-in": "Pull-back",
    "pull-back": "Pull-back",
    "deadhead": "Mid-block deadhead",
    "mid_block_deadhead": "Mid-block deadhead",
    "service": "Revenue service",
}

_TYPE_CSS_COLOR: dict[str, str] = {
    "pull-out": "#28a745",
    "pull-in": "#fd7e14",
    "pull-back": "#fd7e14",
    "deadhead": "#dc3545",
    "mid_block_deadhead": "#dc3545",
    "service": "#0d6efd",
}

_FOLIUM_COLOR: dict[str, str] = {
    "pull-out": "green",
    "pull-in": "orange",
    "pull-back": "orange",
    "deadhead": "red",
    "mid_block_deadhead": "red",
    "service": "blue",
}

# trip_type values that represent a pull-back-to-depot trip
_PULL_BACK_TYPES: frozenset[str] = frozenset({"pull-in", "pull-back"})

# Must match routee-transit's _create_od_key default precision
_OD_KEY_PRECISION = 3

# ---------------------------------------------------------------------------
# Data loading (st.cache_data avoids re-reading on every Streamlit re-run)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_tods(tods_dir: str) -> dict[str, pd.DataFrame]:
    """Load TODS supplement files and depot_metadata.csv from *tods_dir*."""
    p = Path(tods_dir)
    out: dict[str, pd.DataFrame] = {}

    spec: list[tuple[str, str, list[str]]] = [
        ("trips_supplement.txt", "trips", []),
        ("stops_supplement.txt", "stops", ["stop_lat", "stop_lon"]),
        ("stop_times_supplement.txt", "stop_times", ["stop_sequence"]),
        (
            "shapes_supplement.txt",
            "shapes",
            ["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
        ),
        ("depot_metadata.csv", "depot_metadata", []),
    ]

    for fname, key, numeric_cols in spec:
        path = p / fname
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str)
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        out[key] = df

    return out


@st.cache_data(show_spinner=False)
def load_gtfs(gtfs_dir: str) -> dict[str, pd.DataFrame]:
    """Load GTFS files needed for revenue trip display."""
    p = Path(gtfs_dir)
    out: dict[str, pd.DataFrame] = {}

    numeric_map: dict[str, list[str]] = {
        "stops": ["stop_lat", "stop_lon"],
        "shapes": ["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
        "stop_times": ["stop_sequence"],
    }

    for fname, key in [
        ("trips.txt", "trips"),
        ("stop_times.txt", "stop_times"),
        ("shapes.txt", "shapes"),
        ("stops.txt", "stops"),
    ]:
        path = p / fname
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str)
        for col in numeric_map.get(key, []):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        out[key] = df

    return out


@st.cache_data(show_spinner=False)
def load_output(output_dir: str) -> dict[str, pd.DataFrame]:
    """Load routee-transit output files from the parent of the TODS directory.

    Reads ``shapes_final.csv`` (map-matched shapes for revenue trips) and
    ``trip_energy_predictions.csv`` (all trips with shape_id and timing).
    Both files are optional — the app degrades gracefully if absent.
    """
    p = Path(output_dir)
    out: dict[str, pd.DataFrame] = {}

    shapes_p = p / "shapes_final.csv"
    if shapes_p.exists():
        df = pd.read_csv(shapes_p, dtype={"shape_id": str})
        for col in ("shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        out["shapes_final"] = df

    pred_p = p / "trip_energy_predictions.csv"
    if pred_p.exists():
        wanted = {
            "trip_id", "block_id", "shape_id", "trip_type",
            "start_time", "end_time", "route_id", "route_short_name",
        }
        df = pd.read_csv(
            pred_p,
            dtype={"trip_id": str, "block_id": str, "shape_id": str,
                   "route_id": str, "route_short_name": str},
            usecols=lambda c: c in wanted,
        )
        out["trip_predictions"] = df

    return out


@st.cache_data(show_spinner=False)
def load_ntd_for_agency(ntd_id: str) -> pd.DataFrame | None:
    """Return all NTD bus depot facilities for *ntd_id* as a plain DataFrame.

    Uses the bundled NTD facility inventory (2024 Facility Inventory xlsx).
    Returns ``None`` if the file is missing or the package is unavailable.
    """
    try:
        from routee.transit.ntd import load_ntd_facilities

        gdf = load_ntd_facilities(ntd_id=ntd_id)
        # Drop geometry — we use Latitude/Longitude columns directly
        if "geometry" in gdf.columns:
            return pd.DataFrame(gdf.drop(columns=["geometry"]))
        return pd.DataFrame(gdf)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _to_seconds(t: str | None) -> float:
    """HH:MM:SS (possibly >24 h) → total seconds for chronological sorting."""
    if not t or t in ("nan", "None"):
        return float("inf")
    try:
        parts = str(t).strip().split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except Exception:
        return float("inf")


# ---------------------------------------------------------------------------
# Block trip reconstruction
# ---------------------------------------------------------------------------


def _enrich_stop_endpoints(
    trips: list[dict[str, str]],
    tods: dict[str, pd.DataFrame],
    gtfs: dict[str, pd.DataFrame],
) -> None:
    """Fill origin/dest stop IDs and departure/arrival times in-place."""
    trip_ids = {t["trip_id"] for t in trips}
    required = {"trip_id", "stop_id", "stop_sequence"}

    frames: list[pd.DataFrame] = []
    for src in (tods.get("stop_times"), gtfs.get("stop_times")):
        if src is None or not required.issubset(src.columns):
            continue
        keep = ["trip_id", "stop_id", "stop_sequence"]
        for col in ("departure_time", "arrival_time"):
            if col in src.columns:
                keep.append(col)
        filtered = src[src["trip_id"].isin(trip_ids)][keep]
        frames.append(filtered)

    if not frames:
        return

    all_st = pd.concat(frames, ignore_index=True)
    all_st["stop_sequence"] = pd.to_numeric(all_st["stop_sequence"], errors="coerce")
    all_st = all_st.sort_values("stop_sequence")

    def _pick_time(row: pd.Series, prefer: str, fallback: str) -> str:
        for col in (prefer, fallback):
            val = row.get(col)
            if val is not None and not pd.isna(val) and str(val) not in ("", "nan"):
                return str(val)
        return ""

    first = all_st.groupby("trip_id").first().reset_index()
    last = all_st.groupby("trip_id").last().reset_index()

    first_map = {
        row["trip_id"]: {
            "stop_id": str(row.get("stop_id", "") or ""),
            "time": _pick_time(row, "departure_time", "arrival_time"),
        }
        for _, row in first.iterrows()
    }
    last_map = {
        row["trip_id"]: {
            "stop_id": str(row.get("stop_id", "") or ""),
            "time": _pick_time(row, "arrival_time", "departure_time"),
        }
        for _, row in last.iterrows()
    }

    for trip in trips:
        tid = trip["trip_id"]
        if not trip["origin_stop_id"] and tid in first_map:
            trip["origin_stop_id"] = first_map[tid]["stop_id"]
        if not trip["departure_time"] and tid in first_map:
            trip["departure_time"] = first_map[tid]["time"]
        if not trip["dest_stop_id"] and tid in last_map:
            trip["dest_stop_id"] = last_map[tid]["stop_id"]
        if not trip["arrival_time"] and tid in last_map:
            trip["arrival_time"] = last_map[tid]["time"]


def _make_trip(
    trip_id: str,
    trip_type: str,
    shape_id: str = "",
    departure_time: str = "",
    arrival_time: str = "",
) -> dict[str, str]:
    return {
        "trip_id": trip_id,
        "trip_type": trip_type,
        "shape_id": shape_id,
        "departure_time": departure_time,
        "arrival_time": arrival_time,
        "origin_stop_id": "",
        "dest_stop_id": "",
    }


def _safe_str(value: object) -> str:
    """Return str(value) unless value is NA/NaN/None, in which case return ''."""
    if value is None:
        return ""
    try:
        if pd.isna(value):  # type: ignore[call-overload]
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def build_block_trips(
    block_id: str,
    tods: dict[str, pd.DataFrame],
    gtfs: dict[str, pd.DataFrame],
    output: dict[str, pd.DataFrame],
) -> list[dict[str, str]]:
    """Return a chronologically-ordered list of trip dicts for *block_id*.

    Each dict has keys: ``trip_id``, ``trip_type``, ``shape_id``,
    ``departure_time``, ``arrival_time``, ``origin_stop_id``, ``dest_stop_id``.

    The primary data source is ``trip_energy_predictions.csv`` (when present
    in the output directory), which already contains all trips — both revenue
    and deadhead — with shape IDs.  When that file is absent the function falls
    back to GTFS ``trips.txt`` (revenue) plus ``trips_supplement.txt``
    (deadhead).
    """
    trips: list[dict[str, str]] = []

    # Primary path: trip_energy_predictions.csv has all trips with shape_ids
    if "trip_predictions" in output:
        preds = output["trip_predictions"]
        if "block_id" in preds.columns:
            block_rows = preds[preds["block_id"].astype(str) == str(block_id)]
            if not block_rows.empty:
                dedup = block_rows.drop_duplicates(subset=["trip_id"])
                for _, row in dedup.iterrows():
                    trips.append(
                        _make_trip(
                            trip_id=_safe_str(row.get("trip_id")),
                            trip_type=_safe_str(row.get("trip_type")) or "service",
                            shape_id=_safe_str(row.get("shape_id")),
                            departure_time=_safe_str(row.get("start_time")),
                            arrival_time=_safe_str(row.get("end_time")),
                        )
                    )

    # Fallback: GTFS trips.txt + TODS trips_supplement.txt
    if not trips:
        seen: set[str] = set()

        if "trips" in gtfs and "block_id" in gtfs["trips"].columns:
            rev = gtfs["trips"][gtfs["trips"]["block_id"].astype(str) == str(block_id)]
            for _, row in rev.iterrows():
                tid = _safe_str(row.get("trip_id"))
                if tid and tid not in seen:
                    seen.add(tid)
                    trips.append(
                        _make_trip(
                            trip_id=tid,
                            trip_type="service",
                            shape_id=_safe_str(row.get("shape_id")),
                        )
                    )

        if "trips" in tods and "block_id" in tods["trips"].columns:
            dh = tods["trips"][tods["trips"]["block_id"].astype(str) == str(block_id)]
            for _, row in dh.iterrows():
                tid = _safe_str(row.get("trip_id"))
                if tid and tid not in seen:
                    seen.add(tid)
                    tods_type = _safe_str(row.get("TODS_trip_type")) or "deadhead"
                    # Normalise TODS pull-back → internal pull-in
                    if tods_type == "pull-back":
                        tods_type = "pull-in"
                    trips.append(
                        _make_trip(
                            trip_id=tid,
                            # shape_id is not in trips_supplement.txt; resolved later
                            trip_type=tods_type,
                        )
                    )

    if not trips:
        return []

    _enrich_stop_endpoints(trips, tods, gtfs)

    trips.sort(key=lambda t: _to_seconds(t.get("departure_time")))
    return trips


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def get_stop_coords(
    stop_id: str,
    tods: dict[str, pd.DataFrame],
    gtfs: dict[str, pd.DataFrame],
) -> tuple[float, float] | None:
    """Return (lat, lon) for *stop_id*, checking TODS stops then GTFS stops."""
    if not stop_id:
        return None
    for src in (tods.get("stops"), gtfs.get("stops")):
        if src is None:
            continue
        row = src[src["stop_id"].astype(str) == str(stop_id)]
        if row.empty:
            continue
        lat = row.iloc[0].get("stop_lat")
        lon = row.iloc[0].get("stop_lon")
        if pd.notna(lat) and pd.notna(lon):
            return (float(lat), float(lon))
    return None


def get_shape_coords(
    trip: dict[str, str],
    tods: dict[str, pd.DataFrame],
    gtfs: dict[str, pd.DataFrame],
    output: dict[str, pd.DataFrame],
) -> list[tuple[float, float]]:
    """Return ordered (lat, lon) pairs for the trip's shape polyline.

    For revenue trips the map-matched ``shapes_final.csv`` is preferred over
    raw GTFS ``shapes.txt``.  For deadhead trips the shape is looked up in
    ``shapes_supplement.txt`` by shape_id (when available) or by an od_key
    computed from the origin/destination stop coordinates.  A straight-line
    segment between the two endpoints is used as a last resort.
    """
    trip_type = trip["trip_type"]
    shape_id = trip.get("shape_id", "")

    def _lookup(df: pd.DataFrame, sid: str) -> list[tuple[float, float]]:
        rows = df[df["shape_id"].astype(str) == str(sid)]
        if rows.empty:
            return []
        rows = rows.sort_values("shape_pt_sequence")
        return list(zip(rows["shape_pt_lat"].tolist(), rows["shape_pt_lon"].tolist()))

    if trip_type == "service":
        candidates = [output.get("shapes_final"), gtfs.get("shapes")]
        for shapes_df in (df for df in candidates if df is not None):
            if shape_id:
                coords = _lookup(shapes_df, shape_id)
                if coords:
                    return coords
    else:
        shapes = tods.get("shapes")
        if shapes is not None:
            # Direct lookup when shape_id is an od_key (from trip_energy_predictions)
            if shape_id:
                coords = _lookup(shapes, shape_id)
                if coords:
                    return coords

            # Compute od_key from stop coordinates (fallback when shape_id absent)
            origin = get_stop_coords(trip["origin_stop_id"], tods, gtfs)
            dest = get_stop_coords(trip["dest_stop_id"], tods, gtfs)
            if origin and dest:
                od_key = (
                    f"{round(origin[1], _OD_KEY_PRECISION)},"
                    f"{round(origin[0], _OD_KEY_PRECISION)}"
                    f"->"
                    f"{round(dest[1], _OD_KEY_PRECISION)},"
                    f"{round(dest[0], _OD_KEY_PRECISION)}"
                )
                coords = _lookup(shapes, od_key)
                if coords:
                    return coords

    # Final fallback: straight line between endpoints
    origin = get_stop_coords(trip["origin_stop_id"], tods, gtfs)
    dest = get_stop_coords(trip["dest_stop_id"], tods, gtfs)
    if origin and dest:
        return [origin, dest]
    return []


def get_depot_info(
    stop_id: str,
    tods: dict[str, pd.DataFrame],
) -> dict[str, object] | None:
    """Return NTD depot metadata for *stop_id* if it is a depot stop."""
    if not stop_id or not str(stop_id).startswith("depot_"):
        return None
    meta = tods.get("depot_metadata")
    if meta is None:
        return None
    row = meta[meta["stop_id"].astype(str) == str(stop_id)]
    if row.empty:
        return None
    result: dict[str, object] = row.iloc[0].to_dict()  # type: ignore[assignment]
    return result


# ---------------------------------------------------------------------------
# Folium map builder
# ---------------------------------------------------------------------------


def build_folium_map(
    trip: dict[str, str],
    coords: list[tuple[float, float]],
    tods: dict[str, pd.DataFrame],
    gtfs: dict[str, pd.DataFrame],
) -> folium.Map:
    """Return a folium.Map for *trip* with shape polyline, stop markers, and
    an optional depot marker for pull-out / pull-back trips."""
    trip_type = trip["trip_type"]
    css_color = _TYPE_CSS_COLOR.get(trip_type, "#6c757d")

    # Map center
    if coords:
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        center: tuple[float, float] = (sum(lats) / len(lats), sum(lons) / len(lons))
    else:
        fallback = get_stop_coords(trip["origin_stop_id"], tods, gtfs)
        center = fallback or (0.0, 0.0)

    m = folium.Map(location=list(center), zoom_start=13, tiles="CartoDB positron")

    # Shape polyline
    if len(coords) >= 2:
        folium.PolyLine(
            locations=coords,
            color=css_color,
            weight=4,
            opacity=0.85,
            tooltip=f"{_TYPE_LABEL.get(trip_type, trip_type)}: {trip['trip_id']}",
        ).add_to(m)
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    # Origin stop marker (green)
    origin = get_stop_coords(trip["origin_stop_id"], tods, gtfs)
    if origin:
        folium.CircleMarker(
            location=list(origin),
            radius=8,
            color="#28a745",
            fill=True,
            fill_color="#28a745",
            fill_opacity=0.9,
            popup=folium.Popup(
                f"<b>Origin</b><br>Stop: {trip['origin_stop_id']}"
                f"<br>Departs: {trip['departure_time']}",
                max_width=220,
            ),
            tooltip=f"Origin: {trip['origin_stop_id']}",
        ).add_to(m)

    # Destination stop marker (red)
    dest = get_stop_coords(trip["dest_stop_id"], tods, gtfs)
    if dest and dest != origin:
        folium.CircleMarker(
            location=list(dest),
            radius=8,
            color="#dc3545",
            fill=True,
            fill_color="#dc3545",
            fill_opacity=0.9,
            popup=folium.Popup(
                f"<b>Destination</b><br>Stop: {trip['dest_stop_id']}"
                f"<br>Arrives: {trip['arrival_time']}",
                max_width=220,
            ),
            tooltip=f"Destination: {trip['dest_stop_id']}",
        ).add_to(m)

    # Depot marker for pull-out / pull-back (pull-in)
    depot_stop_id: str | None = None
    if trip_type == "pull-out":
        depot_stop_id = trip["origin_stop_id"]
    elif trip_type in _PULL_BACK_TYPES:
        depot_stop_id = trip["dest_stop_id"]

    if depot_stop_id:
        info = get_depot_info(depot_stop_id, tods)
        depot_coords = get_stop_coords(depot_stop_id, tods, gtfs)
        if info and depot_coords:
            name = info.get("Facility Name") or depot_stop_id
            ftype = info.get("Facility Type") or "—"
            agency = info.get("Agency Name") or "—"
            ntd_id = info.get("NTD ID") or "—"

            # Gray markers for every other NTD depot in the same agency
            # Drawn first so the orange selected marker renders on top.
            agency_depots = load_ntd_for_agency(str(ntd_id))
            if agency_depots is not None:
                for _, fac in agency_depots.iterrows():
                    fac_name = str(fac.get("Facility Name") or "")
                    if fac_name == name:
                        continue  # the selected depot — drawn as orange below
                    fac_lat = fac.get("Latitude")
                    fac_lon = fac.get("Longitude")
                    if not (pd.notna(fac_lat) and pd.notna(fac_lon)):
                        continue
                    fac_type = str(fac.get("Facility Type") or "")
                    fac_popup = (
                        f"<b>{fac_name}</b><br>"
                        f"Facility Type: {fac_type}<br>"
                        f"Agency: {agency}<br>"
                        f"NTD ID: {ntd_id}<br>"
                        f"<em>Not the inferred depot</em>"
                    )
                    folium.Marker(
                        location=[float(fac_lat), float(fac_lon)],
                        popup=folium.Popup(fac_popup, max_width=280),
                        tooltip=fac_name,
                        icon=folium.Icon(color="gray", icon="home", prefix="fa"),
                    ).add_to(m)

            # Orange marker for the inferred (selected) depot
            popup_html = (
                f"<b>{name}</b><br>"
                f"NTD Location Type: <em>garage</em><br>"
                f"Facility Type: {ftype}<br>"
                f"Agency: {agency}<br>"
                f"NTD ID: {ntd_id}"
            )
            folium.Marker(
                location=list(depot_coords),
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"Depot (inferred): {name}",
                icon=folium.Icon(color="orange", icon="home", prefix="fa"),
            ).add_to(m)

    return m


# ---------------------------------------------------------------------------
# Streamlit UI helpers
# ---------------------------------------------------------------------------


def _pill(text: str, color: str, highlight: bool = False) -> str:
    """Return an HTML badge pill, optionally outlined to indicate selection."""
    border = (
        f"outline: 3px solid {color}; outline-offset: 3px; transform: scale(1.08);"
        if highlight
        else ""
    )
    return (
        f'<span style="display:inline-block;background:{color};color:#fff;'
        f"padding:4px 10px;border-radius:20px;font-size:0.78rem;"
        f'font-weight:600;{border}">{text}</span>'
    )


def _render_timeline(block_trips: list[dict[str, str]], current_idx: int) -> None:
    """Render a horizontally-scrollable strip of trip-type pills."""
    short = {
        "pull-out": "Pull-out",
        "pull-in": "Pull-back",
        "pull-back": "Pull-back",
        "deadhead": "Deadhead",
        "mid_block_deadhead": "Deadhead",
        "service": "Service",
    }
    pills: list[str] = []
    for i, trip in enumerate(block_trips):
        tt = trip["trip_type"]
        c = _TYPE_CSS_COLOR.get(tt, "#6c757d")
        dep = (trip.get("departure_time") or "")[:5] or "—"
        label = f"{i + 1}. {short.get(tt, tt)}<br><small>{dep}</small>"
        pills.append(_pill(label, c, highlight=(i == current_idx)))

    st.markdown(
        '<div style="display:flex;gap:8px;overflow-x:auto;padding:8px 2px;'
        'scrollbar-width:thin;flex-wrap:nowrap;align-items:center;">'
        + "".join(pills)
        + "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="TODS Block Viewer",
        page_icon="🚌",
        layout="wide",
    )
    st.title("🚌 TODS Block Viewer")
    st.caption(
        "Walk through each trip in a transit block — including inferred deadhead trips — "
        "plotted on an interactive map."
    )

    # ---- Sidebar ----------------------------------------------------------------
    with st.sidebar:
        st.header("Data Sources")

        tods_dir = st.text_input(
            "TODS directory",
            value="reports/saltlake/tods",
            help="Directory containing trips_supplement.txt, shapes_supplement.txt, etc.",
        )
        gtfs_dir = st.text_input(
            "GTFS directory",
            value="routee/transit/resources/sample_inputs/saltlake/gtfs",
            help="Base GTFS feed directory (trips.txt, stops.txt, shapes.txt, stop_times.txt).",
        )

        load_clicked = st.button("Load data", type="primary", use_container_width=True)

        if load_clicked or "loaded" not in st.session_state:
            with st.spinner("Loading TODS files…"):
                st.session_state["tods"] = load_tods(tods_dir)
            with st.spinner("Loading GTFS files…"):
                st.session_state["gtfs"] = load_gtfs(gtfs_dir)
            output_dir = str(Path(tods_dir).parent)
            with st.spinner("Loading output files (if present)…"):
                st.session_state["output"] = load_output(output_dir)
            st.session_state["loaded"] = True
            st.session_state["trip_idx"] = 0
            st.session_state["last_block"] = None

        tods: dict[str, pd.DataFrame] = st.session_state.get("tods", {})
        gtfs: dict[str, pd.DataFrame] = st.session_state.get("gtfs", {})
        output: dict[str, pd.DataFrame] = st.session_state.get("output", {})

        # Status indicators
        has_matched_shapes = "shapes_final" in output
        has_predictions = "trip_predictions" in output
        st.caption(
            ("✅" if has_predictions else "⚠️")
            + " trip_energy_predictions.csv"
            + (" found" if has_predictions else " not found (using raw GTFS)")
        )
        st.caption(
            ("✅" if has_matched_shapes else "⚠️")
            + " shapes_final.csv"
            + (" found" if has_matched_shapes else " not found (using raw GTFS shapes)")
        )

        # Collect block IDs only from TODS output — not from the full GTFS feed.
        # Prefer trip_energy_predictions (all processed blocks) then fall back to
        # trips_supplement (blocks with at least one inferred deadhead trip).
        block_ids: set[str] = set()
        for src_dict, col_key in [
            (output, "trip_predictions"),
            (tods, "trips"),
        ]:
            df = src_dict.get(col_key)
            if df is not None and "block_id" in df.columns:
                block_ids.update(df["block_id"].dropna().astype(str).unique())
            if block_ids:
                break  # stop at the first source that has data

        if not block_ids:
            st.warning("No blocks found. Verify the paths above and click **Load data**.")
            return

        st.divider()
        st.header("Navigation")

        # ---- Route filter --------------------------------------------------------
        # Build route → block mapping from service trips in trip_predictions.
        # Keys are route_id strings; values are display labels (route_short_name
        # when available, otherwise route_id).
        route_label_map: dict[str, str] = {}  # route_id → display label
        route_block_map: dict[str, set[str]] = {}  # route_id → set of block_ids
        preds = output.get("trip_predictions")
        if preds is not None and "route_id" in preds.columns and "trip_type" in preds.columns:
            service = preds[preds["trip_type"] == "service"]
            for route_id, grp in service.groupby("route_id"):
                rid = str(route_id)
                short = grp["route_short_name"].dropna().astype(str)
                short = short[short != "nan"]
                label = short.iloc[0] if not short.empty else rid
                route_label_map[rid] = label
                route_block_map[rid] = set(
                    grp["block_id"].dropna().astype(str).unique()
                ) & block_ids

        _ALL = "__all__"
        if route_label_map:
            route_options = [_ALL] + sorted(
                route_label_map, key=lambda r: route_label_map[r]
            )
            selected_route: str = st.selectbox(
                "Route",
                route_options,
                format_func=lambda r: "All routes" if r == _ALL else route_label_map.get(r, r),
                key="route_select",
            )
            # Reset block + trip when route changes
            if selected_route != st.session_state.get("last_route"):
                st.session_state["last_route"] = selected_route
                st.session_state["last_block"] = None
                st.session_state["trip_idx"] = 0

            filtered_block_ids = (
                block_ids
                if selected_route == _ALL
                else route_block_map.get(selected_route, set())
            )
        else:
            filtered_block_ids = block_ids

        sorted_blocks = sorted(filtered_block_ids, key=lambda x: (len(x), x))
        if not sorted_blocks:
            st.warning("No blocks found for the selected route.")
            return

        selected_block: str = st.selectbox("Block ID", sorted_blocks)

        # Reset trip index when the selected block changes
        if selected_block != st.session_state.get("last_block"):
            st.session_state["trip_idx"] = 0
            st.session_state["last_block"] = selected_block

        block_trips = build_block_trips(selected_block, tods, gtfs, output)

        if not block_trips:
            st.warning(f"No trips found for block **{selected_block}**.")
            return

        n = len(block_trips)
        idx = max(0, min(int(st.session_state.get("trip_idx", 0)), n - 1))

        st.caption(f"Trip **{idx + 1}** of **{n}**")

        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("← Prev", disabled=(idx == 0), use_container_width=True):
                st.session_state["trip_idx"] = idx - 1
                st.rerun()
        with col_next:
            if st.button("Next →", disabled=(idx == n - 1), use_container_width=True):
                st.session_state["trip_idx"] = idx + 1
                st.rerun()

        # Re-read after potential button press
        idx = max(0, min(int(st.session_state.get("trip_idx", 0)), n - 1))
        current_trip = block_trips[idx]

        # ---- Trip details panel --------------------------------------------------
        st.divider()
        st.subheader("Trip Details")

        tt = current_trip["trip_type"]
        label = _TYPE_LABEL.get(tt, tt)
        color = _TYPE_CSS_COLOR.get(tt, "#6c757d")
        st.markdown(_pill(label, color), unsafe_allow_html=True)
        st.write("")

        st.markdown(f"**Trip ID:** `{current_trip['trip_id']}`")
        if current_trip.get("departure_time"):
            st.markdown(f"**Departs:** {current_trip['departure_time']}")
        if current_trip.get("arrival_time"):
            st.markdown(f"**Arrives:** {current_trip['arrival_time']}")
        if current_trip.get("origin_stop_id"):
            st.markdown(f"**Origin stop:** `{current_trip['origin_stop_id']}`")
        if current_trip.get("dest_stop_id"):
            st.markdown(f"**Dest stop:** `{current_trip['dest_stop_id']}`")

        # ---- Depot info (pull-out / pull-back only) ------------------------------
        depot_stop_id: str | None = None
        if tt == "pull-out":
            depot_stop_id = current_trip["origin_stop_id"]
        elif tt in _PULL_BACK_TYPES:
            depot_stop_id = current_trip["dest_stop_id"]

        if depot_stop_id:
            info = get_depot_info(depot_stop_id, tods)
            if info:
                st.divider()
                st.subheader("🏭 Depot")
                name = info.get("Facility Name") or depot_stop_id
                ftype = info.get("Facility Type") or "—"
                agency = info.get("Agency Name") or "—"
                ntd_id = info.get("NTD ID") or "—"
                city = str(info.get("City") or "")
                state_abbr = str(info.get("State") or "")
                location_line = (
                    f"  \n📍 {city}, {state_abbr}" if (city or state_abbr) else ""
                )
                st.info(
                    f"**{name}**  \n"
                    f"NTD Location Type: `garage`  \n"
                    f"Facility Type: {ftype}  \n"
                    f"Agency: {agency}  \n"
                    f"NTD ID: `{ntd_id}`"
                    + location_line
                )

    # ---- Main area --------------------------------------------------------------
    st.subheader(f"Block {selected_block} — {n} trips")
    _render_timeline(block_trips, idx)

    st.divider()

    with st.spinner("Rendering map…"):
        coords = get_shape_coords(current_trip, tods, gtfs, output)
        fmap = build_folium_map(current_trip, coords, tods, gtfs)

    st_folium(fmap, use_container_width=True, height=580, returned_objects=[])


if __name__ == "__main__":
    main()
