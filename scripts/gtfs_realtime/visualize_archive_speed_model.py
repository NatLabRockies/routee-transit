"""Visualize the archive-pipeline speed model (incl. the tuned HGB) on a map.

Counterpart to ``visualize_speed_models.py`` for ``fit_archive_speed_models.py``'s
combined multi-agency output. Filters the combined predictions down to a single
agency, builds a road_id -> geometry lookup from that agency's raw
``link_observations`` parquet (the combined predictions CSVs don't carry
geometry), and reuses ``visualize_speed_models.build_error_map`` to render the
interactive folium map — preferring the tuned HGB model when present.

Usage
-----
    python visualize_archive_speed_model.py --agency dayton
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Make sibling module importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import visualize_speed_models as vsm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
log = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("reports/realtime_archive")
DEFAULT_LINK_OBSERVATIONS_ROOT = Path("gtfsrt_archive_runs/link_observations")


def load_agency_geometry(link_observations_root: Path, agency: str) -> dict[str, str]:
    """Build an agency-prefixed road_id -> WKT geometry lookup from the archive."""
    df = pd.read_parquet(
        link_observations_root,
        columns=["road_id", "geom"],
        filters=[("agency", "=", agency)],
    )
    df = df.dropna(subset=["geom"]).drop_duplicates(subset=["road_id"])
    return {f"{agency}_{rid}": geom for rid, geom in zip(df["road_id"], df["geom"])}


def main(
    results_dir: Path, link_observations_root: Path, agency: str, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(results_dir / "test_predictions.csv")
    all_df = pd.read_csv(results_dir / "all_predictions.csv")
    train_road_ids = set(
        pd.read_csv(results_dir / "train_road_ids.csv")["road_id"].astype(str)
    )

    prefix = f"{agency}_"
    test_df = test_df[test_df["road_id"].astype(str).str.startswith(prefix)].copy()
    all_df = all_df[all_df["road_id"].astype(str).str.startswith(prefix)].copy()
    train_road_ids = {r for r in train_road_ids if r.startswith(prefix)}

    if test_df.empty:
        raise SystemExit(f"No test rows for agency '{agency}' in {results_dir}")

    # Prefer the tuned HGB model (feature engineering + hyperparameter search)
    # when present; build_error_map/_aggregate_roads read this module global.
    vsm.PRIMARY_MODEL = (
        "pred_hgb_tuned" if "pred_hgb_tuned" in test_df.columns else "pred_hgb"
    )
    log.info(
        "Visualizing %s for agency=%s (%d test roads, %d total roads)",
        vsm.PRIMARY_MODEL,
        agency,
        test_df["road_id"].nunique(),
        all_df["road_id"].nunique(),
    )

    geom_lookup = load_agency_geometry(link_observations_root, agency)
    log.info("Geometry lookup: %d roads", len(geom_lookup))

    vsm.build_error_map(test_df, train_road_ids, geom_lookup, output_dir, all_df=all_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agency", required=True, help="Agency slug (as used in archive_speeds.py)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory with fit_archive_speed_models.py outputs "
        f"(default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--link-observations-root",
        type=Path,
        default=DEFAULT_LINK_OBSERVATIONS_ROOT,
        help=f"Root of the link_observations parquet dataset "
        f"(default: {DEFAULT_LINK_OBSERVATIONS_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save the map (default: <results-dir>/maps/<agency>)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.results_dir / "maps" / args.agency)
    main(args.results_dir, args.link_observations_root, args.agency, output_dir)
