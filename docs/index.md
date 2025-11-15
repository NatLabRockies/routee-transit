# RouteE-Transit

RouteE-Transit is a Python package that provides comprehensive tools for predicting energy consumption of transit bus systems. Built on top of NREL's [RouteE-Powertrain](https://github.com/NREL/routee-powertrain) package, RouteE-Transit focuses on transit bus applications, predicting energy consumption for buses based on GTFS data.

The package enables users to work with pre-trained transit bus energy models or their own RouteE-Powertrain models based on real-world telematics data or simulation outputs. RouteE-Transit models predict vehicle energy consumption for transit trips based on factors such as road grade, estimated vehicle speed, and distance.

## Key Features

- **GTFS Integration**: Works with General Transit Feed Specification (GTFS) data to analyze entire transit networks
- **Powertrain Agnostic**: Support for various vehicle types including diesel, hybrid, and battery-electric buses
- **Fleet-wide Analysis**: Predict energy consumption for individual trips, complete bus blocks, or entire bus fleets


## Quickstart
To install RouteE-Transit, see [](installation).

```python
from nrel.routee.transit import GTFSEnergyPredictor

# Create predictor and run complete pipeline
predictor = GTFSEnergyPredictor(
    gtfs_path="path/to/gtfs",
    # depot_path is optional - defaults to NTD depot locations from:
    # https://data.transportation.gov/stories/s/gd62-jzra
)

# Run the complete workflow with a single method call
trip_results = predictor.run(
    vehicle_models="Transit_Bus_Battery_Electric",
    date="2023/08/02",  # Optional, filter to specific date
    routes=["205"],     # Optional, filter to specific routes
    add_hvac=True,      # Include HVAC energy impacts
    output_dir="reports/output",
)
```

For a full example, see [](examples/Utah_Transit_Agency_example). That example can also be run as a script with `python scripts/single_agency_full_analysis.py`.

### Alternative: Step-by-Step Processing

For more control over the workflow, you can invoke each processing step individually:

```python
predictor = GTFSEnergyPredictor(gtfs_path="path/to/gtfs")

# Load and process GTFS data
predictor.load_gtfs_data()
predictor.filter_trips(date="2023/08/02", routes=["205"])

# Add deadhead trips (optional)
predictor.add_mid_block_deadhead()
predictor.add_depot_deadhead()

# Match to road network and add grade
predictor.match_shapes_to_network()
predictor.add_road_grade()

# Predict energy consumption
predictor.predict_energy(
    vehicle_models=["Transit_Bus_Battery_Electric", "Transit_Bus_Diesel"],
    add_hvac=True,
)

# Access results
trip_results = predictor.get_trip_predictions()
link_results = predictor.get_link_predictions()
```


## Available Models
Pretrained transit bus models are included in the RouteE Powertrain package. You can list all available models (including transit buses and other vehicles) with:
```python
import nrel.routee.powertrain as pt

# List all available pre-trained models
print(pt.list_available_models())
```

Each model includes multiple estimators that account for different combinations of features such as speed, road grade, and stop frequency.


