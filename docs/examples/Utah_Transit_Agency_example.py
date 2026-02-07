"""
# Utah Transit Agency Example

In this example, we'll predict the energy consumption for some trips operated by
the Utah Transit Authority (UTA) in Salt Lake City. This requires specifying the
GTFS data we are analyzing, processing it to produce RouteE-Powertrain inputs,
and running a RouteE-Powertrain model to produce energy estimates.

This example uses the `GTFSEnergyPredictor` class, which provides a clean,
extensible API for transit energy prediction.
"""

import logging
import os

from routee.transit import GTFSEnergyPredictor, repo_root

# Set up logging: Clear any existing handlers
logging.getLogger().handlers.clear()

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

# Suppress GDAL/PROJ warnings, which flood the output when we run gradeit
os.environ["PROJ_DEBUG"] = "0"

# Specify input data location
input_directory = repo_root() / "sample-inputs/saltlake/gtfs"
output_directory = repo_root() / "reports/saltlake"

"""
## Quick Start: Using the `run()` Method

For most use cases, the `run()` method provides the simplest way to perform the 
complete energy prediction workflow. This single method call chains together all 
processing steps and returns trip-level energy predictions.

We'll analyze routes 806 and 807 on August 2nd, 2023, using the Battery Electric 
Bus model with HVAC energy impacts included.

Note: depot_path is not specified, so the predictor will use default depot locations
from the National Transit Database's "Public Transit Facilities and Stations - 2023"
dataset (https://data.transportation.gov/stories/s/gd62-jzra).
"""

predictor = GTFSEnergyPredictor(
    gtfs_path=input_directory,
    output_dir=output_directory,
    vehicle_models=["Transit_Bus_Battery_Electric"],
)

trip_results = predictor.run(
    date="2023/08/02",
    routes=["806", "807"],
    add_depot_deadhead=True,
    add_mid_block_deadhead=True,
    add_hvac=True,
    save_results=False,
)

"""
The `run()` method automatically performs all these steps:
1. Loads the GTFS feed
2. Filters trips by date and routes
3. Adds mid-block deadhead trips (between consecutive trips)
4. Adds depot deadhead trips (to/from depot)
5. Matches shapes to OpenStreetMap road network
6. Adds road grade information using USGS elevation data
7. Predicts energy consumption with RouteE-Powertrain
8. Adds estimated HVAC energy impacts
9. Saves results to CSV files

Let's examine the results:
"""

print(trip_results.head())

"""
## Calculate Energy Efficiency Metrics

We can calculate energy efficiency in kWh per mile, including HVAC loads.
The results now include a 'scenario' column (summer/winter) for HVAC impacts.
"""

if "scenario" in trip_results.columns:
    trip_results["kwh_per_mi"] = trip_results["energy_used"] / trip_results["miles"]

print(trip_results.groupby("scenario")["kwh_per_mi"].mean())

"""
## Access Additional Results

After running predictions, you can access link-level results and RouteE inputs:
"""

# Link-level predictions show energy for each road segment
link_results = predictor.get_link_predictions()
print(link_results.head())
