from pathlib import Path

__all__ = [
    "GTFSEnergyPredictor",
    "NetworkRouter",
    "add_HVAC_energy",
]

from .deadhead_router import NetworkRouter
from .predictor import GTFSEnergyPredictor
from .thermal_energy import add_HVAC_energy


def package_root() -> Path:
    """Return the path to the nrel.routee.transit package"""
    return Path(__file__).parent


def repo_root() -> Path:
    """Return the path to the root directory of the repository"""
    return Path(__file__).parent.parent.parent.parent
