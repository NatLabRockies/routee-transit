"""Energy storage capacities declared by the bundled vehicle models."""

from __future__ import annotations

import importlib.resources
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# RouteE-Compass accepts several spellings per energy unit. Normalize them onto
# the labels used by VEHICLE_MODELS so capacities and energy totals compare.
_UNIT_ALIASES: dict[str, str] = {
    "kilowatthours": "kWh",
    "kilowatthour": "kWh",
    "kwh": "kWh",
    "gallonsdiesel": "gallons_diesel",
    "gallonsdieselequivalent": "gallons_diesel",
    "gde": "gallons_diesel",
    "gallonsgasoline": "gallons_gasoline",
    "gallonsgasolineequivalent": "gallons_gasoline",
    "gge": "gallons_gasoline",
    "litersdiesel": "liters_diesel",
    "litersgasoline": "liters_gasoline",
}

VEHICLE_SPEC_COLUMNS = [
    "vehicle",
    "display_name",
    "powertrain",
    "capacity",
    "capacity_unit",
    "usable_capacity_fraction",
    "usable_capacity",
]


def normalize_energy_unit(unit: str) -> str:
    """Map a RouteE-Compass energy unit spelling onto its reporting label.

    Parameters
    ----------
    unit : str
            Unit string as written in a vehicle model config, e.g. ``"kilowatt
            hour"`` or ``"gallons_diesel"``.

    Returns
    -------
    str
            Canonical label such as ``"kWh"`` or ``"gallons_diesel"``.

    Raises
    ------
    ValueError
            If the unit is not one RouteE-Compass recognizes.
    """
    key = unit.strip().lower().replace("_", "").replace(" ", "")
    if key not in _UNIT_ALIASES:
        raise ValueError(f"Unknown energy unit '{unit}'")
    return _UNIT_ALIASES[key]


@dataclass(frozen=True)
class VehicleSpec:
    """Onboard energy storage description for a single vehicle model."""

    name: str
    powertrain: str
    capacity: float
    capacity_unit: str
    usable_capacity_fraction: float
    mass_estimate_lbs: float | None
    display_name: str | None = None

    @property
    def usable_capacity(self) -> float:
        """Capacity available in service, in :attr:`capacity_unit`."""
        return self.capacity * self.usable_capacity_fraction


def _as_float(value: object, name: str, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Vehicle model '{name}' has a non-numeric '{key}': {value!r}")
    return float(value)


def _spec_from_config(config: dict[str, object]) -> VehicleSpec | None:
    name = str(config["name"])
    display_name = config.get("display_name")
    if display_name is not None and (
        not isinstance(display_name, str) or not display_name.strip()
    ):
        raise ValueError(
            f"Vehicle model '{name}' has an invalid 'display_name': {display_name!r}"
        )
    if "battery_capacity" in config:
        capacity = _as_float(config["battery_capacity"], name, "battery_capacity")
        unit = str(config["battery_capacity_unit"])
    elif "fuel_capacity" in config:
        capacity = _as_float(config["fuel_capacity"], name, "fuel_capacity")
        unit = str(config["fuel_capacity_unit"])
    else:
        logger.warning(
            "Vehicle model '%s' declares no energy capacity; "
            "block feasibility will be unavailable for it",
            name,
        )
        return None

    fraction = config.get("usable_capacity_fraction")
    if fraction is None:
        logger.warning(
            "Vehicle model '%s' has no usable_capacity_fraction; "
            "assuming its full capacity is usable",
            name,
        )
    mass = config.get("mass_estimate_lbs")
    return VehicleSpec(
        name=name,
        powertrain=str(config.get("type", "")),
        capacity=capacity,
        capacity_unit=normalize_energy_unit(unit),
        usable_capacity_fraction=1.0
        if fraction is None
        else _as_float(fraction, name, "usable_capacity_fraction"),
        mass_estimate_lbs=None
        if mass is None
        else _as_float(mass, name, "mass_estimate_lbs"),
        display_name=display_name,
    )


def load_vehicle_specs(
    vehicle_models: list[str] | None = None,
) -> dict[str, VehicleSpec]:
    """Read capacity metadata from the bundled vehicle model configs.

    Models shipped by RouteE-Compass rather than RouteE-Transit have no bundled
    config and are therefore absent from the result.

    Parameters
    ----------
    vehicle_models : list of str or None
            Restrict to these model names. If None, all bundled models load.

    Returns
    -------
    dict of str to VehicleSpec
            Specs keyed by vehicle model name.
    """
    src_dir = importlib.resources.files("routee.transit.resources").joinpath(
        "vehicle_models"
    )
    specs: dict[str, VehicleSpec] = {}
    for entry in src_dir.iterdir():
        if not entry.name.endswith(".json"):
            continue
        if vehicle_models is not None and Path(entry.name).stem not in vehicle_models:
            continue
        spec = _spec_from_config(json.loads(entry.read_text(encoding="utf-8")))
        if spec is not None:
            specs[spec.name] = spec
    return specs


def vehicle_specs_table(specs: Mapping[str, VehicleSpec]) -> pd.DataFrame:
    """Flatten vehicle specs into a table for export.

    Parameters
    ----------
    specs : Mapping of str to VehicleSpec
            Specs as returned by :func:`load_vehicle_specs`.

    Returns
    -------
    pd.DataFrame
            One row per vehicle model, sorted by name.
    """
    rows = [
        {
            "vehicle": spec.name,
            "display_name": spec.display_name,
            "powertrain": spec.powertrain,
            "capacity": spec.capacity,
            "capacity_unit": spec.capacity_unit,
            "usable_capacity_fraction": spec.usable_capacity_fraction,
            "usable_capacity": spec.usable_capacity,
        }
        for spec in specs.values()
    ]
    table = pd.DataFrame(rows, columns=VEHICLE_SPEC_COLUMNS)
    return table.sort_values("vehicle", ignore_index=True)
