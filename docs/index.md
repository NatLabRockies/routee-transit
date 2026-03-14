# RouteE-Transit

RouteE-Transit is a Python package for predicting energy consumption of transit bus systems. It uses [RouteE-Compass](https://github.com/NatLabRockies/routee-compass) — NLR's Rust-based routing and energy modeling engine — to process GTFS data and predict energy consumption based on road grade, speed, and distance.

## Key Features

- **GTFS Integration**: Works with General Transit Feed Specification (GTFS) data to analyze entire transit networks
- **Powertrain Agnostic**: Support for various vehicle types including diesel, hybrid, and battery-electric buses
- **Fleet-wide Analysis**: Predict energy consumption for individual trips, complete bus blocks, or entire bus fleets


## Quickstart
To install RouteE-Transit, see [](installation). It takes only a few lines of code to run energy prediction for all trips defined in a GTFS feed:

```python
from routee.transit import GTFSEnergyPredictor

# Create predictor - vehicle_models and output_dir are set here
predictor = GTFSEnergyPredictor(
    gtfs_path="path/to/gtfs",
    vehicle_models=["Transit_Bus_Diesel", "Transit_Bus_Battery_Electric"],
)

# Run the complete workflow with a single method call
trip_results = predictor.run()
```

Plenty of optional inputs allow for filtering down the analysis to a smaller scale. For example, you could include a subset of routes only based on their GTFS `route_short_name`, and only trips on a certain date:

```python
predictor.run(
    date="2023/08/02",
    routes=["806", "807"],
)
```

For a full example, see [](examples/Utah_Transit_Agency_example).

## Available Models
Two pre-trained transit bus models are bundled with RouteE-Transit and are accessed via the `vehicle_models` parameter:

| Model Name | Energy Unit |
|---|---|
| `Transit_Bus_Battery_Electric` | kWh |
| `Transit_Bus_Diesel` | gallons_diesel |

Both models are implemented as RouteE-Compass traversal models in Rust and predict energy based on speed, road grade, and distance.

