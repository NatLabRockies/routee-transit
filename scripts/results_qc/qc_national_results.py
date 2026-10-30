"""Quality-check a national batch run before sharing results.

Reads every dataset's ``results/`` folder under a batch database root and
builds a self-contained HTML report plus small summary CSVs covering:

* coverage (run status by dataset and state),
* agency- and trip-level energy rates by vehicle (kWh-equivalent per mile, so
  all fuels share one axis, plus native units),
* variation by Census region and a map of agency rates,
* variation by day (seasonal HVAC signal for electric buses, service level by
  day of week) and HVAC share,
* deadhead share, and
* rule-based flags for agency/vehicle results that look implausible.

Only pandas, numpy and matplotlib are needed, so it runs on a local copy of
the database without building RouteE-Transit. See this folder's README.

Usage (from the repo root):
    uv run --project scripts/results_qc python scripts/results_qc/qc_national_results.py \
        /path/to/routee-transit-db --out reports/qc_national
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

logger = logging.getLogger("qc_national_results")

KWH_PER_GGE = 33.705

VEHICLE_ORDER = [
    "Transit_Bus_Electric_40ft_300kWh",
    "Transit_Bus_Electric_60ft_600kWh",
    "Transit_Bus_Diesel_40ft",
    "Transit_Bus_Diesel_60ft",
    "Transit_Bus_Hybrid_40ft",
    "Transit_Bus_Hybrid_60ft",
    "Transit_Bus_CNG_40ft",
    "Transit_Bus_CNG_60ft",
]

# Diesel bus of the same length, used to strip service-level changes out of the
# daily electric totals so that only the weather (HVAC) signal is left.
SEASONAL_REFERENCE = {
    "Transit_Bus_Electric_40ft_300kWh": "Transit_Bus_Diesel_40ft",
    "Transit_Bus_Electric_60ft_600kWh": "Transit_Bus_Diesel_60ft",
}

# Rough plausibility windows in kWh-equivalent per mile for service trips. These
# are deliberately loose screening bounds, not validation targets; edit freely.
PLAUSIBLE_KWH_EQ_PER_MILE = {
    "Transit_Bus_Electric_40ft_300kWh": (1.2, 4.0),
    "Transit_Bus_Electric_60ft_600kWh": (1.8, 6.0),
    "Transit_Bus_Diesel_40ft": (5.0, 13.0),
    "Transit_Bus_Diesel_60ft": (7.0, 18.0),
    "Transit_Bus_Hybrid_40ft": (4.0, 11.0),
    "Transit_Bus_Hybrid_60ft": (5.5, 15.0),
    "Transit_Bus_CNG_40ft": (6.0, 16.0),
    "Transit_Bus_CNG_60ft": (8.0, 21.0),
}
MAX_DEADHEAD_SHARE = 0.25
MAX_HVAC_SHARE = 0.40
MAX_INVALID_TRIP_SHARE = 0.01

# Trip-level histograms are accumulated on fixed bins so they stay small.
RATE_BINS = np.round(np.arange(0.0, 30.25, 0.25), 2)

STATE_TO_REGION = {
    **dict.fromkeys(
        [
            "Connecticut",
            "Maine",
            "Massachusetts",
            "New Hampshire",
            "Rhode Island",
            "Vermont",
            "New Jersey",
            "New York",
            "Pennsylvania",
        ],
        "Northeast",
    ),
    **dict.fromkeys(
        [
            "Illinois",
            "Indiana",
            "Michigan",
            "Ohio",
            "Wisconsin",
            "Iowa",
            "Kansas",
            "Minnesota",
            "Missouri",
            "Nebraska",
            "North Dakota",
            "South Dakota",
        ],
        "Midwest",
    ),
    **dict.fromkeys(
        [
            "Delaware",
            "District of Columbia",
            "Florida",
            "Georgia",
            "Maryland",
            "North Carolina",
            "South Carolina",
            "Virginia",
            "West Virginia",
            "Alabama",
            "Kentucky",
            "Mississippi",
            "Tennessee",
            "Arkansas",
            "Louisiana",
            "Oklahoma",
            "Texas",
        ],
        "South",
    ),
    **dict.fromkeys(
        [
            "Arizona",
            "Colorado",
            "Idaho",
            "Montana",
            "Nevada",
            "New Mexico",
            "Utah",
            "Wyoming",
            "California",
            "Oregon",
            "Washington",
        ],
        "West",
    ),
    **dict.fromkeys(["Alaska", "Hawaii"], "Alaska/Hawaii"),
}
REGION_ORDER = [
    "Northeast",
    "Midwest",
    "South",
    "West",
    "Alaska/Hawaii",
    "Multi/Unknown",
]

TRIP_COLUMNS = [
    "trip_id",
    "vehicle",
    "miles",
    "energy_used",
    "energy_unit",
    "mpge",
    "hvac_energy_kWh",
    "trip_count",
    "trip_type",
]
BLOCK_COLUMNS = ["vehicle", "date", "block_id", "energy_used_with_deadhead"]


def load_feed_metadata(db_root: Path) -> pd.DataFrame:
    """Return one row per dataset with feed name, provider, state and region."""
    feeds_path = db_root / "feeds.csv"
    if not feeds_path.exists():
        logger.warning("No feeds.csv in %s; regions will be unknown", db_root)
        return pd.DataFrame(columns=["dataset_id"])
    feeds = pd.read_csv(feeds_path)

    def parse_states(value: Any) -> list[str]:
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parsed = [value]
            return [str(s) for s in parsed if s]
        return []

    feeds["state_list"] = feeds["states"].map(parse_states)
    feeds["state"] = feeds["state_list"].map(
        lambda s: s[0] if len(s) == 1 else ("Multiple" if s else "Unknown")
    )
    feeds["region"] = feeds["state_list"].map(
        lambda s: (
            STATE_TO_REGION.get(s[0], "Multi/Unknown")
            if len({STATE_TO_REGION.get(x) for x in s}) == 1
            else "Multi/Unknown"
        )
    )
    # Mobility DB often leaves feed_name blank, so fall back to the provider.
    feeds["name"] = feeds["name"].fillna(feeds["provider"])
    return feeds.rename(columns={"id": "feed_id", "latest_dataset_id": "dataset_id"})[
        [
            "dataset_id",
            "feed_id",
            "name",
            "provider",
            "state",
            "region",
            "center_latitude",
            "center_longitude",
        ]
    ]


def read_status(results_dir: Path) -> dict[str, Any]:
    status_file = results_dir / "status.json"
    if status_file.exists():
        return dict(json.loads(status_file.read_text()))
    return {}


def summarize_trips(
    trips: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Agency-level rates per vehicle and fixed-bin trip-rate histograms."""
    trips = trips.copy()
    trips["trip_count"] = trips["trip_count"].fillna(0)
    trips["is_service"] = trips["trip_type"].fillna("service").eq("service")
    valid = trips["miles"].gt(0) & trips["mpge"].gt(0) & np.isfinite(trips["mpge"])
    trips["valid"] = valid
    trips["gge"] = np.where(valid, trips["miles"] / trips["mpge"], np.nan)
    trips["kwh_eq_per_mile"] = KWH_PER_GGE / trips["mpge"].where(valid)
    trips["annual_miles"] = trips["miles"] * trips["trip_count"]
    trips["annual_energy"] = trips["energy_used"] * trips["trip_count"]
    trips["annual_gge"] = trips["gge"] * trips["trip_count"]
    trips["annual_hvac"] = trips["hvac_energy_kWh"].fillna(0) * trips["trip_count"]

    rows: list[dict[str, Any]] = []
    hists: dict[str, np.ndarray] = {}
    for vehicle, v in trips.groupby("vehicle"):
        svc = v.loc[v["is_service"] & v["valid"]]
        allv = v.loc[v["valid"]]
        svc_miles = svc["annual_miles"].sum()
        all_miles = allv["annual_miles"].sum()
        row = {
            "vehicle": vehicle,
            "energy_unit": v["energy_unit"].iloc[0],
            "n_service_trips": int(v["is_service"].sum()),
            "n_deadhead_trips": int((~v["is_service"]).sum()),
            "invalid_trip_share": float(1 - v["valid"].mean()),
            "annual_service_miles": svc_miles,
            "annual_deadhead_miles": all_miles - svc_miles,
            "deadhead_share": (all_miles - svc_miles) / all_miles
            if all_miles
            else np.nan,
            "native_per_mile_service": (
                svc["annual_energy"].sum() / svc_miles if svc_miles else np.nan
            ),
            "kwh_eq_per_mile_service": (
                KWH_PER_GGE * svc["annual_gge"].sum() / svc_miles
                if svc_miles
                else np.nan
            ),
            "kwh_eq_per_mile_all": (
                KWH_PER_GGE * allv["annual_gge"].sum() / all_miles
                if all_miles
                else np.nan
            ),
            "hvac_share": (
                v["annual_hvac"].sum() / v["annual_energy"].sum()
                if v["hvac_energy_kWh"].notna().any() and v["annual_energy"].sum()
                else np.nan
            ),
        }
        if not svc.empty:
            weights = svc["trip_count"].to_numpy()
            rates = svc["kwh_eq_per_mile"].to_numpy()
            row["trip_rate_p05"], row["trip_rate_p50"], row["trip_rate_p95"] = (
                weighted_quantiles(rates, weights, [0.05, 0.5, 0.95])
            )
            hists[str(vehicle)] = np.histogram(
                np.clip(rates, RATE_BINS[0], RATE_BINS[-1] - 1e-9),
                bins=RATE_BINS,
                weights=weights,
            )[0]
        rows.append(row)
    return rows, hists


def weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, qs: list[float]
) -> list[float]:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum = np.cumsum(weights)
    if cum[-1] <= 0:
        return [float("nan")] * len(qs)
    return [float(np.interp(q * cum[-1], cum, values)) for q in qs]


def summarize_blocks(blocks: pd.DataFrame) -> pd.DataFrame:
    """Daily totals per vehicle. Exploded route rows repeat block energy, so
    collapse to one row per vehicle, date and block first."""
    blocks = blocks.drop_duplicates(subset=["vehicle", "date", "block_id"])
    return (
        blocks.groupby(["vehicle", "date"])
        .agg(
            energy=("energy_used_with_deadhead", "sum"),
            n_blocks=("block_id", "nunique"),
        )
        .reset_index()
    )


def process_dataset(dataset_dir: Path) -> dict[str, Any]:
    dataset_id = dataset_dir.name
    results_dir = dataset_dir / "results"
    status = read_status(results_dir)
    out: dict[str, Any] = {
        "dataset_id": dataset_id,
        "status": status.get("status", "not_attempted"),
        "includes_deadhead": status.get("includes_deadhead"),
        "failure_type": status.get("failure_type"),
        "error_summary": status.get("error_summary"),
        "agency_rows": [],
        "hists": {},
        "daily": None,
        "error": None,
    }
    trip_path = results_dir / "trip_energy_predictions.csv"
    if not trip_path.exists():
        return out
    try:
        trips = pd.read_csv(
            trip_path,
            usecols=lambda c: c in TRIP_COLUMNS,
            dtype={"trip_id": str},
            low_memory=False,
        )
        for col in TRIP_COLUMNS:
            if col not in trips:
                trips[col] = np.nan
        rows, hists = summarize_trips(trips)
        for row in rows:
            row["dataset_id"] = dataset_id
        out["agency_rows"], out["hists"] = rows, hists

        block_path = results_dir / "block_energy_predictions.csv"
        if block_path.exists():
            blocks = pd.read_csv(
                block_path,
                usecols=BLOCK_COLUMNS,
                dtype={"block_id": str},
                parse_dates=["date"],
            )
            daily = summarize_blocks(blocks)
            daily["dataset_id"] = dataset_id
            out["daily"] = daily
    except Exception as err:  # noqa: BLE001 - report the dataset as unreadable
        out["error"] = f"{type(err).__name__}: {err}"
    return out


def seasonal_index(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily electric/diesel energy ratio, normalized to each agency's mean.

    Diesel energy here does not vary with weather, so the ratio cancels changes
    in how much service runs and leaves the HVAC-driven seasonal signal.
    """
    frames = []
    for electric, diesel in SEASONAL_REFERENCE.items():
        e = daily.loc[daily["vehicle"] == electric, ["dataset_id", "date", "energy"]]
        d = daily.loc[daily["vehicle"] == diesel, ["dataset_id", "date", "energy"]]
        m = e.merge(d, on=["dataset_id", "date"], suffixes=("_e", "_d"))
        m = m.loc[m["energy_d"] > 0]
        m["ratio"] = m["energy_e"] / m["energy_d"]
        m["index"] = m["ratio"] / m.groupby("dataset_id")["ratio"].transform("mean")
        m["vehicle"] = electric
        frames.append(m[["dataset_id", "vehicle", "date", "index"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_flags(agency: pd.DataFrame) -> pd.DataFrame:
    flags: list[dict[str, Any]] = []

    def add(row: pd.Series, check: str, detail: str) -> None:
        flags.append(
            {
                "dataset_id": row["dataset_id"],
                "name": row.get("name"),
                "state": row.get("state"),
                "vehicle": row["vehicle"],
                "check": check,
                "detail": detail,
            }
        )

    for _, row in agency.iterrows():
        lo, hi = PLAUSIBLE_KWH_EQ_PER_MILE.get(row["vehicle"], (0, np.inf))
        rate = row["kwh_eq_per_mile_service"]
        if pd.isna(rate):
            add(row, "no_valid_rate", "no valid service-trip energy")
        elif not lo <= rate <= hi:
            add(row, "rate_out_of_range", f"{rate:.2f} kWh-eq/mi vs {lo}-{hi}")
        # Deadhead miles are the same for every bus, so check one vehicle only.
        if (
            row["vehicle"]
            == agency.loc[agency["dataset_id"] == row["dataset_id"], "vehicle"].iloc[0]
            and row["deadhead_share"] > MAX_DEADHEAD_SHARE
        ):
            add(row, "high_deadhead", f"{row['deadhead_share']:.0%} of miles")
        if row["hvac_share"] > MAX_HVAC_SHARE:
            add(row, "high_hvac", f"{row['hvac_share']:.0%} of energy")
        if row["invalid_trip_share"] > MAX_INVALID_TRIP_SHARE:
            add(
                row,
                "invalid_trips",
                f"{row['invalid_trip_share']:.1%} trips with zero miles or bad MPGe",
            )

    # Physical ordering checks within each agency.
    wide = agency.pivot_table(
        index="dataset_id", columns="vehicle", values="kwh_eq_per_mile_service"
    )
    meta = agency.drop_duplicates("dataset_id").set_index("dataset_id")
    orderings = [
        ("Transit_Bus_Diesel_40ft", "Transit_Bus_Diesel_60ft", "60ft_not_higher"),
        ("Transit_Bus_CNG_40ft", "Transit_Bus_CNG_60ft", "60ft_not_higher"),
        ("Transit_Bus_Hybrid_40ft", "Transit_Bus_Hybrid_60ft", "60ft_not_higher"),
        (
            "Transit_Bus_Electric_40ft_300kWh",
            "Transit_Bus_Electric_60ft_600kWh",
            "60ft_not_higher",
        ),
        ("Transit_Bus_Hybrid_40ft", "Transit_Bus_Diesel_40ft", "hybrid_not_lower"),
        (
            "Transit_Bus_Electric_40ft_300kWh",
            "Transit_Bus_Diesel_40ft",
            "bev_not_lower",
        ),
    ]
    for low, high, check in orderings:
        if low in wide and high in wide:
            bad = wide.loc[wide[low] >= wide[high], [low, high]]
            for ds, low_rate, high_rate in bad.itertuples(name=None):
                flags.append(
                    {
                        "dataset_id": ds,
                        "name": meta.at[ds, "name"] if "name" in meta else None,
                        "state": meta.at[ds, "state"] if "state" in meta else None,
                        "vehicle": f"{low} vs {high}",
                        "check": check,
                        "detail": f"{low_rate:.2f} >= {high_rate:.2f}",
                    }
                )
    return pd.DataFrame(
        flags, columns=["dataset_id", "name", "state", "vehicle", "check", "detail"]
    )


def short(vehicle: str) -> str:
    return vehicle.replace("Transit_Bus_", "").replace("_", " ")


def fig_to_html(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{data}"/>'


def plot_agency_hist(agency: pd.DataFrame, vehicles: list[str]) -> str:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False)
    for ax, vehicle in zip(axes.flat, vehicles):
        v = agency.loc[agency["vehicle"] == vehicle, "kwh_eq_per_mile_service"].dropna()
        ax.hist(v, bins=30, color="#4C72B0")
        lo, hi = PLAUSIBLE_KWH_EQ_PER_MILE.get(vehicle, (None, None))
        if lo is not None:
            ax.axvspan(lo, hi, color="#55A868", alpha=0.12, label="screening range")
        ax.axvline(
            v.median(), color="k", ls="--", lw=1, label=f"median {v.median():.2f}"
        )
        ax.set_title(f"{short(vehicle)} (n={len(v)})", fontsize=10)
        ax.set_xlabel("kWh-eq / mile (service trips)")
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(vehicles) :]:
        ax.axis("off")
    fig.suptitle("Agency-average energy rate, one value per agency")
    return fig_to_html(fig)


def plot_trip_hist(hists: dict[str, np.ndarray], vehicles: list[str]) -> str:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False)
    centers = 0.5 * (RATE_BINS[1:] + RATE_BINS[:-1])
    for ax, vehicle in zip(axes.flat, vehicles):
        h = hists.get(vehicle)
        if h is None or h.sum() == 0:
            ax.axis("off")
            continue
        ax.bar(centers, h / h.sum(), width=RATE_BINS[1] - RATE_BINS[0], color="#8172B2")
        nz = np.nonzero(h)[0]
        ax.set_xlim(
            RATE_BINS[max(nz[0] - 2, 0)], RATE_BINS[min(nz[-1] + 3, len(RATE_BINS) - 1)]
        )
        ax.set_title(short(vehicle), fontsize=10)
        ax.set_xlabel("kWh-eq / mile")
    for ax in list(axes.flat)[len(vehicles) :]:
        ax.axis("off")
    fig.suptitle("Trip-level energy rate, weighted by trips per year (all agencies)")
    return fig_to_html(fig)


def plot_by_region(agency: pd.DataFrame, vehicles: list[str]) -> str:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False)
    for ax, vehicle in zip(axes.flat, vehicles):
        v = agency.loc[agency["vehicle"] == vehicle]
        regions = [r for r in REGION_ORDER if r in set(v["region"])]
        data = [
            v.loc[v["region"] == r, "kwh_eq_per_mile_service"].dropna() for r in regions
        ]
        ax.boxplot(data, showfliers=True, flierprops={"markersize": 3})
        ax.set_xticks(range(1, len(regions) + 1))
        ax.set_xticklabels(
            [f"{r}\n(n={len(d)})" for r, d in zip(regions, data)], fontsize=7
        )
        ax.set_title(short(vehicle), fontsize=10)
        ax.set_ylabel("kWh-eq / mile")
    for ax in list(axes.flat)[len(vehicles) :]:
        ax.axis("off")
    fig.suptitle("Agency-average energy rate by Census region")
    return fig_to_html(fig)


def plot_map(agency: pd.DataFrame, vehicle: str) -> str:
    v = agency.loc[
        (agency["vehicle"] == vehicle)
        & agency["center_longitude"].between(-125, -66)
        & agency["center_latitude"].between(24, 50)
    ]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sc = ax.scatter(
        v["center_longitude"],
        v["center_latitude"],
        c=v["kwh_eq_per_mile_service"],
        s=12
        + 30 * np.sqrt(v["annual_service_miles"] / v["annual_service_miles"].max()),
        cmap="viridis",
        edgecolor="k",
        linewidth=0.3,
    )
    fig.colorbar(sc, ax=ax, label="kWh / mile (service trips)")
    ax.set_title(
        f"{short(vehicle)}: agency rate by location (lower 48; size = annual miles)"
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    return fig_to_html(fig)


def plot_seasonal(season: pd.DataFrame, meta: pd.DataFrame) -> str:
    if season.empty:
        return "<p>No electric/diesel daily block results to compare.</p>"
    s = season.merge(meta[["dataset_id", "region"]], on="dataset_id", how="left")
    s["month"] = s["date"].dt.month
    vehicles = list(s["vehicle"].unique())
    fig, axes = plt.subplots(
        1, len(vehicles), figsize=(7 * len(vehicles), 4.5), squeeze=False
    )
    for ax, vehicle in zip(axes.flat, vehicles):
        v = s.loc[s["vehicle"] == vehicle]
        for region in REGION_ORDER:
            r = v.loc[v["region"] == region]
            if r.empty:
                continue
            # Median across agencies of each agency's monthly mean index.
            monthly = r.groupby(["dataset_id", "month"])["index"].mean().reset_index()
            med = monthly.groupby("month")["index"].median()
            ax.plot(
                med.index,
                med.values,
                marker="o",
                label=f"{region} ({r['dataset_id'].nunique()})",
            )
        ax.axhline(1, color="grey", lw=0.8)
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("month")
        ax.set_ylabel("daily energy vs agency annual mean")
        ax.set_title(short(vehicle), fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(
        "Seasonal signal in electric bus energy (electric / same-size diesel, normalized)"
    )
    return fig_to_html(fig)


def plot_seasonal_spread(season: pd.DataFrame) -> str:
    if season.empty:
        return ""
    amp = (
        season.groupby(["dataset_id", "vehicle"])["index"]
        .agg(lambda x: x.max() - x.min())
        .reset_index(name="amplitude")
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    for vehicle, v in amp.groupby("vehicle"):
        ax.hist(v["amplitude"], bins=30, alpha=0.6, label=short(str(vehicle)))
    ax.set_xlabel("max - min of daily index over the year")
    ax.set_ylabel("agencies")
    ax.set_title("How much electric energy swings day to day within each agency")
    ax.legend(fontsize=8)
    return fig_to_html(fig)


def plot_day_of_week(daily: pd.DataFrame) -> str:
    d = daily.loc[daily["vehicle"] == "Transit_Bus_Diesel_40ft"].copy()
    if d.empty:
        return ""
    d["dow"] = d["date"].dt.dayofweek
    d["rel"] = d["energy"] / d.groupby("dataset_id")["energy"].transform("max")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([d.loc[d["dow"] == i, "rel"] for i in range(7)], showfliers=False)
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_ylabel("daily energy / agency's peak day")
    ax.set_title("Service level by day of week (Diesel 40ft daily totals)")
    return fig_to_html(fig)


def plot_share_hists(agency: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ref = agency.loc[
        agency["vehicle"] == "Transit_Bus_Diesel_40ft", "deadhead_share"
    ].dropna()
    axes[0].hist(ref, bins=40, color="#C44E52")
    axes[0].axvline(
        MAX_DEADHEAD_SHARE, color="k", ls="--", lw=1, label="flag threshold"
    )
    axes[0].set_xlabel("deadhead share of annual miles")
    axes[0].set_ylabel("agencies")
    axes[0].set_title("Deadhead share of miles")
    axes[0].legend(fontsize=8)
    for vehicle in SEASONAL_REFERENCE:
        h = agency.loc[agency["vehicle"] == vehicle, "hvac_share"].dropna()
        axes[1].hist(h, bins=40, alpha=0.6, label=short(vehicle))
    axes[1].axvline(MAX_HVAC_SHARE, color="k", ls="--", lw=1, label="flag threshold")
    axes[1].set_xlabel("HVAC share of annual energy")
    axes[1].set_title("HVAC share (electric buses)")
    axes[1].legend(fontsize=8)
    return fig_to_html(fig)


def rate_table(agency: pd.DataFrame, vehicles: list[str]) -> pd.DataFrame:
    rows = []
    for vehicle in vehicles:
        v = agency.loc[agency["vehicle"] == vehicle]
        kwh = v["kwh_eq_per_mile_service"].dropna()
        native = v["native_per_mile_service"].dropna()
        unit = v["energy_unit"].iloc[0] if not v.empty else ""
        total_miles = v["annual_service_miles"].sum()
        national = (
            (v["kwh_eq_per_mile_service"] * v["annual_service_miles"]).sum()
            / total_miles
            if total_miles
            else np.nan
        )
        rows.append(
            {
                "vehicle": short(vehicle),
                "agencies": len(kwh),
                "kWh-eq/mi p05": kwh.quantile(0.05),
                "p25": kwh.quantile(0.25),
                "median": kwh.median(),
                "p75": kwh.quantile(0.75),
                "p95": kwh.quantile(0.95),
                "mile-weighted national": national,
                "median native per mile": (
                    f"{native.median():.3f} {unit}" if len(native) else ""
                ),
                "median MPGe": KWH_PER_GGE / kwh.median() if len(kwh) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def html_table(df: pd.DataFrame) -> str:
    return df.to_html(float_format=lambda x: f"{x:,.2f}", index=False, border=0)


def build_report(
    out_dir: Path,
    db_root: Path,
    status: pd.DataFrame,
    agency: pd.DataFrame,
    hists: dict[str, np.ndarray],
    daily: pd.DataFrame,
    season: pd.DataFrame,
    flags: pd.DataFrame,
    meta: pd.DataFrame,
) -> Path:
    vehicles = [v for v in VEHICLE_ORDER if v in set(agency["vehicle"])] + sorted(
        set(agency["vehicle"]) - set(VEHICLE_ORDER)
    )
    status_counts = (
        status["status"]
        .value_counts()
        .rename_axis("status")
        .reset_index(name="datasets")
    )
    by_region = (
        agency.drop_duplicates("dataset_id")["region"]
        .value_counts()
        .reindex(REGION_ORDER)
        .dropna()
        .astype(int)
        .rename_axis("region")
        .reset_index(name="agencies with results")
    )
    flag_counts = (
        flags.groupby("check").size().rename("count").reset_index()
        if not flags.empty
        else pd.DataFrame(columns=["check", "count"])
    )
    sections = [
        "<h1>RouteE-Transit national results QC</h1>",
        f"<p>Database: <code>{db_root}</code></p>",
        "<h2>1. Coverage</h2>",
        status_counts.pipe(html_table),
        by_region.pipe(html_table),
        "<h2>2. Energy rate distributions</h2>",
        (
            "<p>Rates are for service (revenue) trips, weighted by how many days each "
            "trip runs. kWh-equivalent uses 33.705 kWh per gallon of gasoline "
            "equivalent so every fuel shares one axis. Green bands are loose "
            "screening ranges set at the top of the script, not validation targets.</p>"
        ),
        rate_table(agency, vehicles).pipe(html_table),
        plot_agency_hist(agency, vehicles),
        plot_trip_hist(hists, vehicles),
        "<h2>3. Variation by region</h2>",
        plot_by_region(agency, vehicles),
        plot_map(agency, "Transit_Bus_Electric_40ft_300kWh"),
        plot_map(agency, "Transit_Bus_Diesel_40ft"),
        "<h2>4. Variation by day</h2>",
        (
            "<p>In this pipeline each GTFS shape gets one energy value per bus, so "
            "non-HVAC energy is the same every day a trip runs. Day-to-day change "
            "in per-mile energy comes only from HVAC on electric buses (typical-"
            "year weather). Expect cold- and hot-climate agencies to swing more and "
            "mild coastal ones less. Diesel, hybrid and CNG daily totals change "
            "only with the amount of service.</p>"
        ),
        plot_seasonal(season, meta),
        plot_seasonal_spread(season),
        plot_day_of_week(daily),
        "<h2>5. Deadhead and HVAC shares</h2>",
        (
            "<p>Known issue #114: agencies whose depots could not be geocoded fell "
            "back to the nearest national depot, which inflates deadhead.</p>"
        ),
        plot_share_hists(agency),
        "<h2>6. Flags</h2>",
        flag_counts.pipe(html_table),
        "<p>Full list in <code>flags.csv</code>. First 200 shown.</p>",
        flags.head(200).pipe(html_table),
    ]
    style = (
        "<style>body{font-family:system-ui,sans-serif;max-width:1400px;margin:24px auto;"
        "padding:0 16px;color:#222}table{border-collapse:collapse;margin:8px 0 16px;"
        "font-size:13px}td,th{padding:3px 10px;border-bottom:1px solid #ddd;text-align:right}"
        "th{background:#f3f3f3}img{max-width:100%;margin:8px 0}</style>"
    )
    path = out_dir / "qc_report.html"
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>National results QC"
        f"</title>{style}</head><body>{''.join(sections)}</body></html>"
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db_root", type=Path, help="Batch database root")
    parser.add_argument("--out", type=Path, default=Path("reports/qc_national"))
    parser.add_argument("--n_processes", type=int, default=min(mp.cpu_count(), 32))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    dataset_dirs = sorted(
        p for p in args.db_root.iterdir() if p.is_dir() and (p / "results").is_dir()
    )
    logger.info("Reading %d dataset result folders", len(dataset_dirs))
    with mp.Pool(args.n_processes) as pool:
        results = pool.map(process_dataset, dataset_dirs, chunksize=4)

    meta = load_feed_metadata(args.db_root)
    status = pd.DataFrame(
        [
            {
                k: r[k]
                for k in (
                    "dataset_id",
                    "status",
                    "includes_deadhead",
                    "failure_type",
                    "error_summary",
                    "error",
                )
            }
            for r in results
        ]
    ).merge(meta, on="dataset_id", how="left")
    status.to_csv(args.out / "dataset_status.csv", index=False)
    unreadable = status.loc[status["error"].notna()]
    if not unreadable.empty:
        logger.warning("%d datasets had unreadable results", len(unreadable))

    agency = pd.DataFrame([row for r in results for row in r["agency_rows"]])
    agency = agency.merge(meta, on="dataset_id", how="left")
    agency["region"] = agency["region"].fillna("Multi/Unknown")
    agency.to_csv(args.out / "agency_vehicle_summary.csv", index=False)

    hists: dict[str, np.ndarray] = {}
    for r in results:
        for vehicle, h in r["hists"].items():
            hists[vehicle] = hists.get(vehicle, 0) + h
    pd.DataFrame({"bin_lo": RATE_BINS[:-1], "bin_hi": RATE_BINS[1:], **hists}).to_csv(
        args.out / "trip_rate_histograms.csv", index=False
    )

    daily_frames = [r["daily"] for r in results if r["daily"] is not None]
    daily = (
        pd.concat(daily_frames, ignore_index=True)
        if daily_frames
        else pd.DataFrame(
            columns=["vehicle", "date", "energy", "n_blocks", "dataset_id"]
        )
    )
    daily.to_csv(args.out / "daily_energy_by_agency.csv.gz", index=False)
    season = seasonal_index(daily) if not daily.empty else pd.DataFrame()

    flags = build_flags(agency)
    flags.to_csv(args.out / "flags.csv", index=False)

    report = build_report(
        args.out, args.db_root, status, agency, hists, daily, season, flags, meta
    )
    logger.info("Wrote %s", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
