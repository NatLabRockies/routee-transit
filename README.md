# routee-transit
Application of RouteE Powertrain to transit bus energy prediction using GFTS data as inputs.

## Setup
We use Pixi for dependency management and packaging. To install Pixi, see [the documentation](https://pixi.sh/latest/).

Once you have Pixi installed, from the root directory (`routee-transit`), install the environment with `pixi install`. If you encounter GDAL-related errors when installing on a Mac, try `brew install gdal` and then `pixi install` again.

This will create a virtual environment based on the dependencies described in `pyproject.toml`. To execute code in this virtual environment, use `pixi run <my_file.py>`, or `pixi shell` to run all subsequent commands in that environment. To use the development environment, add the `-e dev` flag, e.g., `pixi shell -e dev`.

You can also set up your environment using `pip`, if preferred. In your virtual environment, simply run `pip install .`

## Running the current pipeline
`scripts/single_agency_full_analysis.py` provides an example of running the full pipeline to predict energy consumption for some or all trips in an agency's GTFS feed.

