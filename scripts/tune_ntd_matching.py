"""Tune NTD fuzzy matching parameters against labelled test data.

Reads the hand-labelled test CSV (tests/ntd-agencies-test.csv) and performs a
grid search over the key matching parameters to maximise accuracy. Rows with
empty ``ntd_agency_id`` are excluded from accuracy scoring (no ground-truth).

The expensive WRatio, IDF, and geodesic distance computations are done once
up-front; only the weight/threshold parameters are varied during the grid
search, making it fast.

Usage:
    pixi run -e dev-py311 python scripts/tune_ntd_matching.py
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from geopy.distance import geodesic
from rapidfuzz.fuzz import WRatio

from routee.transit.ntd import (
    _compute_token_idf,
    _idf_query_coverage_score,
    _load_ntd_agencies,
    _tokenize_name,
)

TEST_CSV = Path(__file__).resolve().parent.parent / "tests" / "ntd-agencies-test.csv"


# ---------------------------------------------------------------------------
# Precompute all scores for each (test_agency, ntd_candidate) pair
# ---------------------------------------------------------------------------


def precompute_scores(
    test_df: pd.DataFrame,
    agencies: pd.DataFrame,
    token_idf: dict[str, float],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
]:
    """Precompute WRatio, IDF, distance, and agency-ID-match matrices.

    Returns
    -------
    wratio_matrix : ndarray of shape (n_test, n_candidates)
    idf_matrix : ndarray of shape (n_test, n_candidates)
    distance_matrix : ndarray of shape (n_test, n_candidates)  [km]
    id_match_matrix : ndarray of shape (n_test, n_candidates)  bool
    """
    n_test = len(test_df)
    n_cand = len(agencies)

    official_names = agencies["Agency_Name"].fillna("").tolist()
    common_names = agencies["Common_Name"].fillna("").tolist()
    cand_lats = agencies["Lat"].values
    cand_lons = agencies["Lon"].values
    cand_ntd_ids = agencies["NTD_ID"].values

    wratio_matrix = np.zeros((n_test, n_cand))
    idf_matrix = np.zeros((n_test, n_cand))
    distance_matrix = np.zeros((n_test, n_cand))
    id_match_matrix = np.zeros((n_test, n_cand), dtype=bool)

    for i, (_, row) in enumerate(test_df.iterrows()):
        agency_name = row["agency_name"]
        lat = row["center_latitude"]
        lon = row["center_longitude"]
        query_tokens = _tokenize_name(agency_name)

        # Agency ID matching
        raw_id = row.get("agency_id")
        if raw_id is not None and str(raw_id).strip():
            try:
                padded_id = str(int(raw_id)).zfill(5)
                id_match_matrix[i] = cand_ntd_ids == padded_id
            except (ValueError, TypeError):
                pass

        # WRatio scores
        official_scores = np.array(
            [WRatio(agency_name, name) for name in official_names]
        )
        common_scores = np.array([WRatio(agency_name, name) for name in common_names])
        wratio_matrix[i] = np.maximum(official_scores, common_scores)

        # IDF scores
        official_idf = np.array(
            [
                _idf_query_coverage_score(query_tokens, name, token_idf)
                for name in official_names
            ]
        )
        common_idf = np.array(
            [
                _idf_query_coverage_score(query_tokens, name, token_idf)
                for name in common_names
            ]
        )
        idf_matrix[i] = np.maximum(official_idf, common_idf)

        # Geodesic distances
        query_point = (lat, lon)
        distance_matrix[i] = np.array(
            [
                geodesic(query_point, (cand_lats[j], cand_lons[j])).km
                for j in range(n_cand)
            ]
        )

    return wratio_matrix, idf_matrix, distance_matrix, id_match_matrix


# ---------------------------------------------------------------------------
# Fast evaluation using precomputed matrices
# ---------------------------------------------------------------------------


def evaluate_params(
    wratio_matrix: NDArray[np.float64],
    idf_matrix: NDArray[np.float64],
    distance_matrix: NDArray[np.float64],
    id_match_matrix: NDArray[np.bool_],
    expected_ids: list[str],
    candidate_ids: NDArray[np.str_],
    *,
    wratio_weight: float,
    idf_weight: float,
    name_threshold: float,
    combined_name_weight: float,
    combined_proximity_weight: float,
    proximity_scale_km: float,
    max_distance_km: float,
    id_bonus: float,
) -> tuple[int, int, list[dict[str, str | None]]]:
    """Evaluate a parameter set against precomputed scores."""
    n_test = wratio_matrix.shape[0]
    correct = 0
    total = n_test
    details: list[dict[str, str | None]] = []

    # Vectorised name scores: (n_test, n_candidates)
    name_scores = wratio_weight * wratio_matrix + idf_weight * idf_matrix

    for i in range(n_test):
        expected_id = expected_ids[i]

        # Apply name threshold mask
        mask = name_scores[i] >= name_threshold
        if not mask.any():
            details.append(
                {
                    "idx": str(i),
                    "expected": expected_id,
                    "predicted": None,
                    "correct": "✗",
                }
            )
            continue

        cand_name_scores = name_scores[i][mask]
        cand_distances = distance_matrix[i][mask]
        cand_ids = candidate_ids[mask]
        cand_id_matches = id_match_matrix[i][mask]

        # Exponential proximity decay
        proximity = np.exp(-cand_distances / proximity_scale_km)
        combined = (
            combined_name_weight * (cand_name_scores / 100.0)
            + combined_proximity_weight * proximity
        )

        # Agency ID bonus
        combined[cand_id_matches] += id_bonus

        best_pos = int(combined.argmax())
        best_distance = cand_distances[best_pos]

        if best_distance > max_distance_km:
            predicted_id: str | None = None
        else:
            predicted_id = str(cand_ids[best_pos])

        is_correct = predicted_id == expected_id
        if is_correct:
            correct += 1

        details.append(
            {
                "idx": str(i),
                "expected": expected_id,
                "predicted": predicted_id,
                "correct": "✓" if is_correct else "✗",
            }
        )

    return correct, total, details


def main() -> None:
    print("Loading test data and NTD agencies...")
    test_df = pd.read_csv(TEST_CSV, dtype={"ntd_agency_id": str})
    agencies = _load_ntd_agencies(bus_only=False)
    token_idf = _compute_token_idf(agencies)

    # Filter to rows with expected matches only
    has_expected = test_df["ntd_agency_id"].notna() & (
        test_df["ntd_agency_id"].str.strip() != ""
    )
    n_skipped = int((~has_expected).sum())
    test_df_labelled = test_df[has_expected].reset_index(drop=True)
    n_labelled = len(test_df_labelled)
    print(f"Test cases with expected NTD match: {n_labelled}")
    print(f"Test cases without expected match (skipped): {n_skipped}")
    print()

    # Normalise expected IDs
    expected_ids = [str(eid).zfill(5) for eid in test_df_labelled["ntd_agency_id"]]
    candidate_ids: NDArray[np.str_] = np.asarray(agencies["NTD_ID"], dtype=str)

    # Precompute expensive scoring matrices
    print("Precomputing WRatio, IDF, distance, and agency-ID scores...")
    t0 = time.time()
    wratio_matrix, idf_matrix, distance_matrix, id_match_matrix = precompute_scores(
        test_df_labelled, agencies, token_idf
    )
    print(f"Precomputation done in {time.time() - t0:.1f}s")
    print()

    # --- Grid search ---
    param_grid: dict[str, list[float]] = {
        "wratio_weight": [0.5, 0.7, 0.9],
        "idf_weight": [0.0, 0.1, 0.3, 0.5],
        "name_threshold": [20, 30, 40, 50],
        "combined_name_weight": [0.5, 0.6, 0.7, 0.8],
        "combined_proximity_weight": [0.2, 0.3, 0.4, 0.5],
        "proximity_scale_km": [50, 75, 100, 150],
        "max_distance_km": [200, 300, 400, 600],
        "id_bonus": [0.0, 0.3, 0.5],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    print(f"Evaluating {len(combos):,} parameter combinations...")
    start = time.time()

    best_correct = -1
    best_params: dict[str, float] = {}
    best_details: list[dict[str, str | None]] = []

    for combo in combos:
        params = dict(zip(keys, combo))
        correct, total, details = evaluate_params(
            wratio_matrix,
            idf_matrix,
            distance_matrix,
            id_match_matrix,
            expected_ids,
            candidate_ids,
            wratio_weight=params["wratio_weight"],
            idf_weight=params["idf_weight"],
            name_threshold=params["name_threshold"],
            combined_name_weight=params["combined_name_weight"],
            combined_proximity_weight=params["combined_proximity_weight"],
            proximity_scale_km=params["proximity_scale_km"],
            max_distance_km=params["max_distance_km"],
            id_bonus=params["id_bonus"],
        )
        if correct > best_correct:
            best_correct = correct
            best_params = params
            best_details = details

    elapsed = time.time() - start
    print(f"Grid search completed in {elapsed:.1f}s")
    print()

    # --- Report best results ---
    total = n_labelled
    print("=" * 70)
    print(f"BEST ACCURACY: {best_correct}/{total} ({100 * best_correct / total:.1f}%)")
    print("=" * 70)
    print()
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print()

    # --- Current defaults comparison ---
    print("-" * 70)
    print("Comparison with CURRENT defaults:")
    print("-" * 70)
    current_correct, _, current_details = evaluate_params(
        wratio_matrix,
        idf_matrix,
        distance_matrix,
        id_match_matrix,
        expected_ids,
        candidate_ids,
        wratio_weight=0.5,
        idf_weight=0.0,
        name_threshold=20,
        combined_name_weight=0.5,
        combined_proximity_weight=0.2,
        proximity_scale_km=75,
        max_distance_km=200,
        id_bonus=0.3,
    )
    print(
        f"  Current accuracy: {current_correct}/{total} ({100 * current_correct / total:.1f}%)"
    )
    print(
        f"  Best accuracy:    {best_correct}/{total} ({100 * best_correct / total:.1f}%)"
    )
    print()

    # --- Per-agency details for best params ---
    agency_names = test_df_labelled["agency_name"].tolist()
    print("-" * 70)
    print("Per-agency results (best params):")
    print("-" * 70)
    print(f"{'Status':<6} {'Agency Name':<45} {'Expected':<8} {'Predicted':<8}")
    print("-" * 70)
    for d in best_details:
        idx = int(d["idx"])  # type: ignore[arg-type]
        print(
            f"{d['correct']:<6} {agency_names[idx]:<45} "
            f"{d['expected']:<8} {d['predicted'] or 'None':<8}"
        )
    print()

    # --- Mismatches with current defaults ---
    mismatches = [
        (int(d["idx"]), d)  # type: ignore[arg-type]
        for d in current_details
        if d["correct"] == "✗"
    ]
    if mismatches:
        print("-" * 70)
        print(f"Mismatches with CURRENT defaults ({len(mismatches)}):")
        print("-" * 70)
        print(f"{'Agency Name':<45} {'Expected':<8} {'Predicted':<8}")
        for idx, d in mismatches:
            print(
                f"{agency_names[idx]:<45} "
                f"{d['expected']:<8} {d['predicted'] or 'None':<8}"
            )


if __name__ == "__main__":
    main()
