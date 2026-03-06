from pathlib import Path

__all__ = [
    "GTFSEnergyPredictor",
    "create_deadhead_shapes",
    "add_HVAC_energy",
    "sample_inputs_path",
    "depot_path",
]

from .deadhead_router import create_deadhead_shapes
from .predictor import GTFSEnergyPredictor
from .thermal_energy import add_HVAC_energy


def package_root() -> Path:
    """Return the path to the nrel.routee.transit package"""
    return Path(__file__).parent


def sample_inputs_path() -> Path:
    """Return the path to the sample inputs directory"""
    return package_root() / "resources" / "sample_inputs"


def depot_path() -> Path:
    """Return the path to the FTA_Depot directory"""
    return sample_inputs_path() / "FTA_Depot"
