from pathlib import Path

__all__ = [
    "GTFSEnergyPredictor",
    "build_routee_features_with_osm",
    "predict_for_all_trips",
    "aggregate_results_by_trip",
    "add_HVAC_energy",
]

# New object-oriented interface
from .prediction.predictor import GTFSEnergyPredictor

# Legacy functional interface (maintained for backward compatibility)
from .prediction.add_temp_feature import add_HVAC_energy
from .prediction.gtfs_feature_processing import build_routee_features_with_osm
from .prediction.routee import aggregate_results_by_trip, predict_for_all_trips


def package_root() -> Path:
    """Return the path to the nrel.routee.powertrain package"""
    return Path(__file__).parent


def repo_root() -> Path:
    """Return the path to the root directory of the repository"""
    return Path(__file__).parent.parent.parent.parent
