"""Helpers for assembling the RouteE-Compass transit-energy dataset config.

These functions generate and patch the ``transit_energy.toml`` config and copy
RouteE-Transit's bundled custom vehicle models into a generated Compass
dataset. They are independent of GTFS and operate purely on the Compass output
directory.
"""

import importlib.resources
import json
import logging
import shutil
from pathlib import Path

import tomlkit

logger = logging.getLogger(__name__)


def copy_transit_config(
    output_directory: Path, vehicle_models: list[str] | None = None
) -> None:
    """
    Copy the transit_energy.toml from package resources to the output directory.

    Args:
        output_directory: Directory where the Compass dataset is being written.
        vehicle_models: Optional list of vehicle models to include. If None, all are included.
    """
    config_text = (
        importlib.resources.files("routee.transit.resources")
        .joinpath("transit_energy.toml")
        .read_text(encoding="utf-8")
    )
    config = tomlkit.loads(config_text)

    # Filter vehicle_models if requested
    if vehicle_models is not None:
        vehicle_set = set(vehicle_models)
        search = config.get("search", {})
        traversal = search.get("traversal", {})
        models = traversal.get("models", [])
        for model in models:
            if model.get("type") == "transit_energy" and "vehicle_input_files" in model:
                model["vehicle_input_files"] = [
                    p
                    for p in model["vehicle_input_files"]
                    if Path(p).stem in vehicle_set
                ]

    output_path = output_directory / "transit_energy.toml"
    with open(output_path, "w") as f:
        tomlkit.dump(config, f)
    logger.info(f"Copied transit_energy.toml to {output_path}")


def read_configured_vehicle_models(config_path: Path) -> set[str]:
    """
    Return the set of vehicle model names referenced in a transit_energy.toml.

    Reads the ``vehicle_input_files`` of the ``transit_energy`` traversal model
    and returns the stem of each referenced vehicle JSON (i.e. the model name).

    Args:
        config_path: Path to a generated ``transit_energy.toml``.
    """
    config = tomlkit.loads(config_path.read_text(encoding="utf-8"))
    models = config.get("search", {}).get("traversal", {}).get("models", [])
    names: set[str] = set()
    for model in models:
        if model.get("type") == "transit_energy":
            for p in model.get("vehicle_input_files", []):
                names.add(Path(p).stem)
    return names


def copy_custom_vehicle_models(
    output_directory: Path, vehicle_models: list[str] | None = None
) -> None:
    """
    Copy RouteE-Transit's bundled custom vehicle models into the generated
    Compass dataset.

    The standard transit models (``Transit_Bus_Battery_Electric``,
    ``Transit_Bus_Diesel``) ship with RouteE-Compass and are copied into
    ``vehicles/`` and ``models/`` by the POWERTRAIN generation phase. Custom
    models bundled with RouteE-Transit (under
    ``routee/transit/resources/vehicle_models/``) are not, so this function
    copies each requested custom vehicle JSON and its referenced ``.bin`` into
    the dataset's ``vehicles/`` directory (the JSON references the ``.bin`` by
    filename in the same directory).

    Args:
        output_directory: Directory where the Compass dataset is being written.
        vehicle_models: Optional list of vehicle models to include. If None, all
            bundled custom models are copied.
    """
    src_dir = importlib.resources.files("routee.transit.resources").joinpath(
        "vehicle_models"
    )
    vehicles_out = output_directory / "vehicles"
    vehicles_out.mkdir(exist_ok=True)

    for entry in src_dir.iterdir():
        if not entry.name.endswith(".json"):
            continue
        model_name = Path(entry.name).stem
        if vehicle_models is not None and model_name not in vehicle_models:
            continue

        config = json.loads(entry.read_text(encoding="utf-8"))
        shutil.copy(str(entry), vehicles_out / entry.name)

        bin_name = Path(config["model_input_file"]).name
        bin_src = src_dir.joinpath(bin_name)
        shutil.copy(str(bin_src), vehicles_out / bin_name)
        logger.info(f"Copied custom vehicle model '{model_name}' to {vehicles_out}")
