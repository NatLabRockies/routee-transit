import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
from unittest.mock import MagicMock, patch

from fit_poc_speed_model import aggregate_agency, balanced_weights, make_manifest
from export_speed_model_onnx import build_inference_matrix
from routee.transit.speed_model import (
    TIME_BIN_FEATURES,
    compute_stop_density,
    compute_transit_speed_features,
)


def observations() -> pd.DataFrame:
    timestamps = [
        "2026-09-18 03:00:00",  # Thursday 20:00 Pacific
        "2026-09-19 15:00:00",  # Saturday 08:00 Pacific
        "2026-09-18 13:00:00",  # Friday 06:00 Pacific
    ]
    return pd.DataFrame(
        {
            "road_id": ["r"] * 9,
            "first_timestamp": np.repeat(timestamps, 3),
            "mph_moving": [10.0] * 9,
            "n_observations": [2] * 9,
            "speed_source": ["observed"] * 9,
            "highway": ["residential"] * 9,
            "maxspeed_mph": [25.0] * 9,
            "lanes": [2.0] * 9,
            "grade": [0.0] * 9,
            "grade_abs": [0.0] * 9,
            "link_length_km": [0.2] * 9,
            "n_stops": [1.0] * 9,
        }
    )


def test_local_bins_and_training_inference_features() -> None:
    agg = aggregate_agency(observations(), "test", "America/Los_Angeles")
    assert set(zip(agg["time_bin"], agg["is_weekday"])) == {(4, 1), (1, 0), (1, 1)}
    assert (agg[TIME_BIN_FEATURES].sum(axis=1) == 1).all()
    assert (agg["mph_moving_mean"] == 10.0).all()
    manifest = make_manifest(agg)
    assert "scheduled_speed_mph" not in manifest["input_order"]
    matrix = build_inference_matrix(agg, manifest)
    assert matrix.shape == (3, len(manifest["input_order"]))
    assert np.isfinite(matrix).all()


def test_agency_weights_are_equal() -> None:
    frame = pd.DataFrame(
        {
            "agency": ["a", "a", "b"],
            "total_observations": [10, 30, 1000],
        }
    )
    weights = balanced_weights(frame)
    assert weights[:2].sum() == pytest.approx(weights[2])
    assert weights[1] / weights[0] == pytest.approx(3)


def test_invalid_timestamps_fail() -> None:
    frame = observations()
    frame.loc[0, "first_timestamp"] = None
    with pytest.raises(ValueError, match="timestamps"):
        aggregate_agency(frame, "test", "America/Los_Angeles")


def test_stop_density_bounds_and_missing_values() -> None:
    density = compute_stop_density(
        pd.Series([1.0, 1.0, 0.0, 1.0, 1.0, np.nan]),
        pd.Series([0.2, 0.001, 0.2, 0.0, np.nan, 0.2]),
    )
    np.testing.assert_allclose(density, [5.0, 20.0, 0.0, np.nan, np.nan, np.nan])


def test_stop_density_manifest_and_inference() -> None:
    agg = aggregate_agency(observations(), "test", "America/Los_Angeles")
    agg["stop_density_per_km"] = compute_stop_density(
        agg["n_stops"], agg["link_length_km"]
    )
    manifest = make_manifest(agg, include_stop_density=True)
    matrix = build_inference_matrix(agg, manifest)
    index = manifest["input_order"].index("stop_density_per_km")
    np.testing.assert_array_equal(matrix[:, index], 5.0)
    assert "stop_density_per_km_was_missing" in manifest["input_order"]

    edges = gpd.GeoDataFrame(
        {
            "edge_id": [0, 1],
            "maxspeed": [25.0, 25.0],
            "lanes": [2, 2],
            "grade": [0.0, 0.0],
            "grade_abs": [0.0, 0.0],
            "distance": [200.0, 1.0],
            "highway": ["residential", "residential"],
        }
    )
    with patch("routee.transit.speed_model._compute_n_stops", return_value=np.ones(2)):
        features = compute_transit_speed_features(edges, MagicMock(), manifest)
    np.testing.assert_array_equal(features["stop_density_per_km"], [5.0, 20.0])
