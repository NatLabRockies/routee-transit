"""
# End-to-End Test: Utah Transit Agency

This test verifies the energy consumption prediction workflow for the Utah
Transit Authority (UTA) in Salt Lake City. It covers GTFS data processing,
map matching, and energy estimation using RouteE-Compass.
"""

import logging
import os
import tempfile
import pandas as pd
import geopandas as gpd
from routee.transit import GTFSEnergyPredictor, repo_root

# Set up logging
logging.getLogger().handlers.clear()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

# Suppress GDAL/PROJ warnings
os.environ["PROJ_DEBUG"] = "0"


def test_e2e_uta() -> None:
    """
    Test the full energy prediction workflow for a subset of UTA routes.
    """
    # Specify input data location
    input_directory = repo_root() / "sample-inputs/saltlake/gtfs"

    # use a temp directory for output
    output_directory = tempfile.mkdtemp()

    # Initialize predictor
    predictor = GTFSEnergyPredictor(
        gtfs_path=input_directory,
        output_dir=output_directory,
        vehicle_models=["Transit_Bus_Battery_Electric"],
        overwrite=True,
    )

    # Run energy prediction
    trip_results = predictor.run(
        date="2023/08/02",
        routes=["806", "205"],
        add_depot_deadhead=True,
        add_mid_block_deadhead=True,
        add_hvac=True,
        save_results=False,
    )

    # Assertions for trip results
    assert isinstance(trip_results, pd.DataFrame), (
        "trip_results should be a pandas DataFrame"
    )
    assert not trip_results.empty, "trip_results should not be empty"

    expected_columns = [
        "trip_id",
        "miles",
        "energy_used",
        "energy_unit",
        "start_time",
        "trip_duration_minutes",
    ]
    for col in expected_columns:
        assert col in trip_results.columns, f"Missing expected column: {col}"

    assert (trip_results["energy_unit"] == "kWh").all(), (
        "Energy unit should be kWh for electric bus"
    )

    print(trip_results.describe())

    # Verify HVAC energy calculation if present
    if "scenario" in trip_results.columns:
        winter_results = trip_results[trip_results["scenario"] == "winter"]
        summer_results = trip_results[trip_results["scenario"] == "summer"]

        assert not winter_results.empty, "Winter scenario results should not be empty"
        assert not summer_results.empty, "Summer scenario results should not be empty"

        assert (winter_results["energy_used"] > 0).all(), (
            "Winter energy should be positive"
        )
        assert (summer_results["energy_used"] > 0).all(), (
            "Summer energy should be positive"
        )

    # Link-level predictions
    link_results = predictor.get_link_predictions()
    if (
        not isinstance(link_results, gpd.GeoDataFrame)
        and "geometry" in link_results.columns
    ):
        link_results = gpd.GeoDataFrame(
            link_results, geometry="geometry", crs="EPSG:4326"
        )

    assert isinstance(link_results, gpd.GeoDataFrame), (
        "link_results should be a GeoDataFrame"
    )
    assert not link_results.empty, "link_results should not be empty"
    assert "energy_used" in link_results.columns, (
        "link_results should contain energy_used column from CompassApp"
    )

    # Matched shapes (map matching result)
    matched_shapes = predictor.matched_shapes
    assert isinstance(matched_shapes, (pd.DataFrame, gpd.GeoDataFrame)), (
        "matched_shapes should be a pandas DataFrame or GeoDataFrame"
    )
    assert not matched_shapes.empty, "matched_shapes should not be empty"


if __name__ == "__main__":
    test_e2e_uta()
    print("E2E Test passed successfully!")
