from pathlib import Path

__all__ = [
    "GTFSEnergyPredictor",
    "NetworkRouter",
    "build_routee_features_with_osm",
    "add_HVAC_energy",
]

# New object-oriented interface
from .predictor import GTFSEnergyPredictor
from .deadhead_router import NetworkRouter

# Legacy functional interface (maintained for backward compatibility)
from .thermal_energy import add_HVAC_energy
from .gtfs_processing import build_routee_features_with_osm


def package_root() -> Path:
    """Return the path to the nrel.routee.transit package"""
    return Path(__file__).parent


def repo_root() -> Path:
    """Return the path to the root directory of the repository"""
    return Path(__file__).parent.parent.parent.parent
