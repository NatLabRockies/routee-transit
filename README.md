# <img src="docs/images/routeelogo.png" alt="Routee Powertrain" width="100"/>

# RouteE-Transit (`nrel.routee.transit`)
RouteE-Transit provides a complete pipeline for predicting the energy consumption of transit bus trips based on GTFS data and a [RouteE-Powertrain](https://github.com/NREL/routee-powertrain) model. The package matches GTFS shapes to the OpenStreetMap road network, aggregates speed, distance, and grade estimates at the OSM road link level, and then uses a trained RouteE-Powertrain model to predict energy consumption.

## Quick Start

```python
from nrel.routee.transit import GTFSEnergyPredictor

# Create predictor and run complete pipeline
predictor = GTFSEnergyPredictor(
    gtfs_path="path/to/gtfs",
    # depot_path is optional - defaults to NTD depot locations
)

# Run analysis with a single method call
trip_results = predictor.run(
    vehicle_models="Transit_Bus_Battery_Electric",
    date="2023/08/02",  # Optional: filter to specific date
    routes=["205"],     # Optional: filter to specific routes
    add_hvac=True,      # Include HVAC energy impacts
)
```

See the [documentation](https://nrel.github.io/routee-transit/) for more examples and details.

## Setup
### Using Pixi (recommended for developers)
Install Pixi following the instructions in [its documentation](https://pixi.sh/latest/).

Once you have Pixi installed, from the root directory (`routee-transit`), install the environment with `pixi install`. If you encounter GDAL-related errors when installing on a Mac, try `brew install gdal` and then `pixi install` again.

This will create a virtual environment based on the dependencies described in `pyproject.toml`. To execute code in this virtual environment, use `pixi run <my_file.py>`, or `pixi shell` to run all subsequent commands in that environment. To use the development environment, add the `-e dev` flag, e.g., `pixi shell -e dev`.

### Using pip
You can also set up your environment using `pip`, if preferred. Create a new virtual environment using the tool of your choice (e.g., `conda create -n routee-transit` / `conda activate routee-transit` if using conda). In your virtual environment, run `pip install .` from the root directory. To include development dependencies, use `pip install ".[dev]"`.

## Key Features

- **Simple API**: Use the `run()` method for most workflows, or call individual methods for fine-grained control
- **HVAC Energy**: Includes seasonal HVAC energy impacts for battery-electric buses
- **Deadhead Trips**: Models both mid-block deadhead (between trips) and depot deadhead (pull-out/pull-in)
- **Multiple Models**: Predict for multiple vehicle types in a single workflow

## Running Examples
`scripts/single_agency_full_analysis.py` shows how to use RouteE-Transit to predict energy consumption for some or all trips in an agency's GTFS feed. See also `docs/examples/Utah_Transit_Agency_example.py` for a documented example.


## Building and Viewing Documentation
To build the documentation, you must have the `dev` dependencies installed. 

If you are using `pixi`, there is a pixi task defined in `pyproject.toml` you can use:

```bash
pixi run docs
```

If you installed RouteE-Transit with `pip`, you can build the docs with:

```bash
python docs/examples/_convert_examples.py  # create .ipynb file from example for jupyter-book
jupyter-book build docs  # build the docs pages
```

Be aware that `jupyter-book` will run all of the documentation examples when the docs are built, which can take a few minutes. To build the documentation faster, you can skip the examples by simply commenting them out in the table of contents (`docs/_toc.yml`).

