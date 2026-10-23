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
    vehicle_models=["Transit_Bus_Diesel_40ft", "Transit_Bus_Electric_40ft_300kWh"],
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

By default, route filtering works at the **trip level**, so individual trips on the requested routes are always included even if their block also serves other routes. If you enable deadhead trip estimation, filtering automatically switches to **block level** to ensure complete blocks (see [](prediction) for details):

```python
predictor.run(
    date="2023/08/02",
    routes=["806", "807"],
    add_mid_block_deadhead=True,
    add_depot_deadhead=True,
)
```

For a full example, see [](examples/Utah_Transit_Agency_example).

## Available Models
Eight pre-trained transit bus models are bundled with RouteE-Transit and are selected via the `vehicle_models` parameter:

| Model Name | Powertrain | Bus Length | Reported Energy Unit |
|---|---|---|---|
| `Transit_Bus_Electric_40ft_300kWh` | Battery electric (300 kWh pack) | 40 ft | kWh |
| `Transit_Bus_Electric_60ft_600kWh` | Battery electric (600 kWh pack) | 60 ft | kWh |
| `Transit_Bus_Diesel_40ft` | Diesel | 40 ft | gallons_diesel |
| `Transit_Bus_Diesel_60ft` | Diesel | 60 ft | gallons_diesel |
| `Transit_Bus_Hybrid_40ft` | Diesel hybrid | 40 ft | gallons_diesel |
| `Transit_Bus_Hybrid_60ft` | Diesel hybrid | 60 ft | gallons_diesel |
| `Transit_Bus_CNG_40ft` | Compressed natural gas | 40 ft | kWh |
| `Transit_Bus_CNG_60ft` | Compressed natural gas | 60 ft | kWh |

If `vehicle_models` is omitted, every supported model is run. All bundled models are
[RouteE-Powertrain](https://github.com/NatLabRockies/routee-powertrain) models evaluated
inside RouteE-Compass, predicting energy from link speed and road grade. See
[](prediction) for details on how they are applied.

