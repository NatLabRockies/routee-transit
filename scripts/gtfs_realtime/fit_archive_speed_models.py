"""Fit speed prediction models on the gtfsrt.io archive link-observations dataset.

Counterpart to ``fit_speed_models.py`` for the archive-based pipeline
(``archive_speeds.py``). Reads the model-ready, hive-partitioned parquet
dataset it produces —
``link_observations/agency=<slug>/date=<date>/part.parquet`` — applies the
same cleaning/outlier-removal steps as the JSONL-scrape workflow, then reuses
its aggregation + model-fitting logic (``fit_and_evaluate_models``) so the two
pipelines stay in sync without duplicating modeling code.

Usage
-----
    # Every agency present in the dataset
    python fit_archive_speed_models.py \\
        --link-observations-root gtfsrt_archive_runs/link_observations

    # A subset
    python fit_archive_speed_models.py \\
        --link-observations-root gtfsrt_archive_runs/link_observations \\
        --agency dayton,mountainline,citybus-lafayette
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

# Make sibling module importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_speed_models import (
    SPEED_CEIL_MPH,
    SPEED_FLOOR_MPH,
    TARGET,
    HyperParamValue,
    add_temporal_features,
    fit_and_evaluate_models,
    remove_outliers,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_LINK_OBSERVATIONS_ROOT = Path("gtfsrt_archive_runs/link_observations")
DEFAULT_OUTPUT_DIR = Path("reports/realtime_archive")


def discover_agencies(root: Path) -> list[str]:
    """List agency slugs present in the link-observations dataset."""
    return sorted(
        d.name.split("=", 1)[1]
        for d in root.iterdir()
        if d.is_dir() and d.name.startswith("agency=")
    )


def load_and_clean_archive(root: Path, agency: str) -> pd.DataFrame:
    """Load one agency's link observations and apply the same filters as
    :func:`fit_speed_models.load_and_clean`, adapted for the archive parquet
    schema (``agency``/``date`` are already columns, not derived from a path).
    """
    log.info("Loading %s from %s", agency, root)
    df = pd.read_parquet(root, filters=[("agency", "=", agency)])
    df["agency"] = df["agency"].astype(str)
    log.info("  Raw rows: %d", len(df))

    df = df.dropna(subset=[TARGET])
    df = df[np.isfinite(df[TARGET].to_numpy(dtype=float))]
    # Keep only directly observed links (>=2 GPS pings on the link)
    if "speed_source" in df.columns:
        df = df[df["speed_source"] == "observed"]
    # Hard floor/ceiling
    df = df[(df[TARGET] >= SPEED_FLOOR_MPH) & (df[TARGET] <= SPEED_CEIL_MPH)]
    log.info("  After basic filters: %d rows", len(df))

    # road_ids are only unique within an agency's OSM network — prefix them so
    # they don't collide when training across agencies (same convention as
    # fit_speed_models.load_and_clean).
    df["road_id"] = agency + "_" + df["road_id"].astype(str)
    return cast(pd.DataFrame, df)


def main(
    root: Path,
    agencies: list[str],
    output_dir: Path,
    tune_hgb: bool = True,
    tune_n_iter: int = 25,
    tune_cv_splits: int = 4,
    tuned_model_keys: list[str] | None = None,
    tuned_fixed_params: dict[str, dict[str, HyperParamValue]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs: list[pd.DataFrame] = []
    for agency in agencies:
        df_agency = load_and_clean_archive(root, agency)
        df_agency = remove_outliers(df_agency)
        df_agency = add_temporal_features(df_agency)
        all_dfs.append(df_agency)

    df = pd.concat(all_dfs, ignore_index=True)
    log.info(
        "Combined data from %d agencies (%s): %d rows",
        len(agencies),
        ", ".join(agencies),
        len(df),
    )

    fit_and_evaluate_models(
        df,
        agencies,
        output_dir,
        tune_hgb=tune_hgb,
        tune_n_iter=tune_n_iter,
        tune_cv_splits=tune_cv_splits,
        tuned_model_keys=tuned_model_keys,
        tuned_fixed_params=tuned_fixed_params,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--link-observations-root",
        type=Path,
        default=DEFAULT_LINK_OBSERVATIONS_ROOT,
        help=f"Root of the link_observations parquet dataset "
        f"(default: {DEFAULT_LINK_OBSERVATIONS_ROOT})",
    )
    parser.add_argument(
        "--agency",
        default="all",
        help="Comma-separated agency slugs, or 'all' (default) for every "
        "agency present under --link-observations-root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write model outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-tune-hgb",
        dest="tune_hgb",
        action="store_false",
        help="Skip the tuned HGB model (feature engineering + hyperparameter search)",
    )
    parser.add_argument(
        "--tune-n-iter",
        type=int,
        default=25,
        help="Number of RandomizedSearchCV samples for HGB tuning (default: 25)",
    )
    parser.add_argument(
        "--tune-cv-splits",
        type=int,
        default=4,
        help="Number of spatial GroupKFold splits for HGB tuning (default: 4)",
    )
    parser.add_argument(
        "--tuned-models",
        default=None,
        help="Comma-separated subset of tuned models to fit: hgb,rf,gbr "
        "(default: all three)",
    )
    parser.add_argument(
        "--fixed-params-json",
        default=None,
        help="JSON dict of {model_key: {param: value}} to skip the "
        "RandomizedSearchCV step and refit directly with known params, e.g. "
        '\'{"rf": {"n_estimators": 300, "max_depth": 8}}\'',
    )
    args = parser.parse_args()

    if args.agency == "all":
        agencies = discover_agencies(args.link_observations_root)
        if not agencies:
            parser.error(
                f"no agency=<slug> directories found under "
                f"{args.link_observations_root}"
            )
    else:
        agencies = [a.strip() for a in args.agency.split(",") if a.strip()]

    main(
        args.link_observations_root,
        agencies,
        args.output_dir,
        tune_hgb=args.tune_hgb,
        tune_n_iter=args.tune_n_iter,
        tune_cv_splits=args.tune_cv_splits,
        tuned_model_keys=args.tuned_models.split(",") if args.tuned_models else None,
        tuned_fixed_params=json.loads(args.fixed_params_json)
        if args.fixed_params_json
        else None,
    )
