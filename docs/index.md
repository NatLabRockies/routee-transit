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
from nrel.routee.transit import (
    build_routee_features_with_osm,
    predict_for_all_trips,
)

# Build features for predicting with RouteE-Powertrain
routee_input_df = build_routee_features_with_osm(
    input_directory="path/to/gtfs",
    add_road_grade=True,  # Add elevation difference
)

# Run a RouteE-Powertrain model
routee_results = predict_for_all_trips(
    routee_input_df=routee_input_df,
    routee_vehicle_model="Transit_Bus_Diesel",
)
```

For a full example, see [](examples/Utah_Transit_Agency_example). That example can also be run as a script with `python scripts/single_agency_full_analysis.py`.


## Available Models
Pretrained transit bus models are included in the RouteE Powertrain package. You can list all available models (including transit buses and other vehicles) with:
```python
import nrel.routee.powertrain as pt

# List all available pre-trained models
print(pt.list_available_models())
```

Each model includes multiple estimators that account for different combinations of features such as speed, road grade, and stop frequency.


