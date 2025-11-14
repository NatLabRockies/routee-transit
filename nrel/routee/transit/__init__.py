from pathlib import Path

__all__ = [
    "GTFSEnergyPredictor",
    "NetworkRouter",
    "add_HVAC_energy",
]

# New object-oriented interface
from .predictor import GTFSEnergyPredictor
from .deadhead_router import NetworkRouter

# Legacy functional interface (maintained for backward compatibility)
from .thermal_energy import add_HVAC_energy


def package_root() -> Path:
    """Return the path to the nrel.routee.transit package"""
    return Path(__file__).parent


def repo_root() -> Path:
    """Return the path to the root directory of the repository"""
    return Path(__file__).parent.parent.parent.parent
