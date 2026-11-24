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
    vehicle_models=["Transit_Bus_Electric_40ft_300kWh"],
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

Let's examine the results. The columns include information about the energy predictions made (including any HVAC component) as well as GTFS fields that apply to each trip, and the date whose weather was used:
"""
trip_results.columns
trip_results[["trip_id", "vehicle", "date", "energy_used", "miles"]].head()
"""
## Analyze Energy Efficiency

We can calculate energy efficiency in kWh per mile, including HVAC loads.
"""
trip_results["kwh_per_mi"] = trip_results["energy_used"] / trip_results["miles"]
"""
### Efficiency by Route
How does typical energy efficiency compare between the two routes?

We can check by filtering out deadhead trips and then grouping by route:
"""
# Only include revenue service trips
service_results = trip_results[trip_results["trip_type"] == "service"].copy()
service_results.groupby("route_short_name")["kwh_per_mi"].mean().sort_values(
    ascending=False
)
"""
Route 807 requires more energy on average.

### Deadhead Energy
Deadhead trips are non-revenue movements: between consecutive trips in a block
(`mid_block_deadhead`) and to and from the depot (`pull-out` / `pull-in`). We can see
how much energy they add relative to revenue service:
"""
trip_results.groupby("trip_type")["energy_used"].sum()
"""
## Access Additional Results

Besides trip-level results, you can also access link-level results. These detailed results can help you better understand differences in predictions across trips.
"""
# Link-level predictions show energy for each road segment
link_results = predictor.get_link_predictions()
link_results.head()
