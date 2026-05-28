from pathlib import Path

__all__ = [
    "GTFSEnergyPredictor",
    "create_deadhead_shapes",
    "add_HVAC_energy",
    "write_tods_deadhead",
    "sample_inputs_path",
    "ntd_path",
    "load_ntd_facilities",
    "match_agency_to_ntd",
    "NTDAgencyMatch",
]

from .deadhead_router import create_deadhead_shapes
from .ntd import (
    NTDAgencyMatch,
    load_ntd_facilities,
    match_agency_to_ntd,
)
from .predictor import GTFSEnergyPredictor
from .thermal_energy import add_HVAC_energy
from .tods_export import write_tods_deadhead


def package_root() -> Path:
    """Return the path to the routee.transit package"""
    return Path(__file__).parent


def sample_inputs_path() -> Path:
    """Return the path to the sample inputs directory"""
    return package_root() / "resources" / "sample_inputs"


def ntd_path() -> Path:
    """Return the path to the bundled NTD resources directory"""
    return package_root() / "resources" / "ntd"
