from pathlib import Path

__all__ = [
    "build_routee_features_with_osm",
    "predict_for_all_trips",
    "aggregate_results_by_trip",
]

from .prediction.gtfs_feature_processing import build_routee_features_with_osm
from .prediction.routee import predict_for_all_trips, aggregate_results_by_trip


def package_root() -> Path:
    """Return the path to the nrel.routee.powertrain package"""
    return Path(__file__).parent


def repo_root() -> Path:
    """Return the path to the root directory of the repository"""
    return Path(__file__).parent.parent.parent.parent
