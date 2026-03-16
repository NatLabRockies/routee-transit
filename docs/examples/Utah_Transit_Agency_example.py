"""
# Utah Transit Authority

In this example, we'll predict the energy consumption for some trips operated by
the Utah Transit Authority (UTA) in Salt Lake City. This requires specifying the
GTFS data we are analyzing, map-matching it to the road network, and running a
RouteE-Compass model to produce energy estimates.

This example uses the `GTFSEnergyPredictor` class, which provides a clean,
extensible API for transit energy prediction.
"""

from pathlib import Path
from routee.transit import GTFSEnergyPredictor, sample_inputs_path

# Specify input data location
input_directory = sample_inputs_path() / "saltlake/gtfs"
output_directory = Path("reports/saltlake")
"""
## Quick Start: Using the `run()` Method

For most use cases, the `run()` method provides the simplest way to perform the 
complete energy prediction workflow. This single method call chains together all 
processing steps and returns trip-level energy predictions.

We'll analyze routes 806 and 807 on August 2nd, 2023, using the Battery Electric 
Bus model. We'll include deadhead trips and estimated HVAC energy.
"""
predictor = GTFSEnergyPredictor(
    gtfs_path=input_directory,
    vehicle_models=["Transit_Bus_Battery_Electric"],
    output_dir=output_directory,
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
5. Matches shapes to OpenStreetMap road network and adds road grade (via RouteE-Compass)
6. Predicts energy consumption with RouteE-Compass
7. Adds estimated HVAC energy impacts
8. Saves results to CSV files

Let's examine the results. The columns include information about the energy predictions made (including any HVAC component) as well as GTFS fields that apply to each trip, and a description of the weather scenario considered:
"""
trip_results.columns
trip_results[["trip_id", "vehicle", "scenario", "energy_used", "miles"]].head()
"""
## Analyze Energy Efficiency

We can calculate energy efficiency in kWh per mile, including HVAC loads.
The results include a `scenario` column (summer/winter.median) for HVAC impacts.

### Weather Impacts
Let's see how energy efficiency in kWh/mi compares across different weather scenarios:
"""
if "scenario" in trip_results.columns:
    trip_results["kwh_per_mi"] = trip_results["energy_used"] / trip_results["miles"]

trip_results.groupby("scenario")["kwh_per_mi"].mean()
"""
We can see that winter requires the greatest energy, since the cold climate in Utah requires a heavy HVAC load.

### Efficiency by Route
How does typical energy efficiency in the `median` scenario (meaning temperatures are taken from the day of a typical year with the median average temperature)?

We can check by filtering by scenario and then grouping by route:
"""
# First, filter out any deadhead trips
median_results = trip_results[trip_results["trip_type"] == "service"].copy()
# Then, only include median weather impacts
median_results = median_results[median_results["scenario"] == "median"]
median_results.groupby("route_short_name")["kwh_per_mi"].mean().sort_values(
    ascending=False
)
"""
Route 807 requires more energy on average.
"""
"""
## Access Additional Results

Besides trip-level results, you can also access link-level results. These detailed results can help you better understand differences in predictions across trips.
"""
# Link-level predictions show energy for each road segment
link_results = predictor.get_link_predictions()
link_results.head()
