"""
# Utah Transit Agency Example
In this example, we'll predict the energy consumption for some trips operated by the Utah Transit Authority (UTA) in Salt Lake City. This requires specifying the GTFS data we are analyzing, processing it to produce RouteE-Powertrain inputs, and running a RouteE-Powertrain model to produce energy estimates.
"""

import logging
import multiprocessing as mp
import os

from nrel.routee.transit import (
    build_routee_features_with_osm,
    predict_for_all_trips,
    repo_root,
)

# Set up logging: Clear any existing handlers
logging.getLogger().handlers.clear()

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

# Suppress GDAL/PROJ warnings, which flood the output when we run gradeit
os.environ["PROJ_DEBUG"] = "0"
# Set inputs
n_proc = mp.cpu_count()
input_directory = repo_root() / "sample-inputs/saltlake"
output_directory = repo_root() / "reports/saltlake"
if not output_directory.exists():
    output_directory.mkdir(parents=True)
"""
## Process GTFS Data into RouteE Inputs
`build_routee_features_with_osm()` analyzes a GTFS feed to prepare input features for energy prediction with RouteE-Powertrain. It performs the following steps:
- Upsamples all shapes so they are suitable for map matching
- Uses NREL's `mappymatch` package to match each shape to a set of OpenStreetMap road links.
- Uses NREL's `gradeit` package to add estimated average grade to each road link. USGS elevation tiles are downloaded and cached if needed.
"""
routee_input_df = build_routee_features_with_osm(
    input_directory=input_directory,
    n_trips=30,  # make predictions for 30 randomly sampled trips
    add_road_grade=True,
    n_processes=n_proc,
)
"""
The output of `build_routee_features_with_osm()` is a DataFrame where each row represents the traversal of a particular road network edge during a particular bus trip. It includes the features needed to make energy predictions with RouteE, such as the travel time reported by OpenStreetMap (`travel_time_osm`), the distance (`distances_ft`), and the estimated road grade as a decimal value (`grade_dec_unfiltered`/`grade_dec_filtered`, depending on whether filtering is used in `gradeit`).
"""
routee_input_df.head()
"""
## Predict Energy Consumption with RouteE-Powertrain
We can now make energy predictions with the data in `routee_input_df` and any trained RouteE Powertrain model. We'll use `"Transit_Bus_Battery_Electric"`, included in `nrel.routee.powertrain` 1.3.2, which is trained on real-world energy data from an electric bus in Salt Lake City.

`predict_with_all_trips()` provides a convenient wrapper for making energy consumption predictions given a RouteE model and the input variables necessary to predict with it:
"""
routee_vehicle_model = "Transit_Bus_Battery_Electric"
routee_results = predict_for_all_trips(
    routee_input_df=routee_input_df,
    routee_vehicle_model=routee_vehicle_model,
    n_processes=n_proc,
)
"""
`routee_results` contains link-level energy predictions for each trip.
"""
routee_results.head()
"""
We can aggregate over trip IDs to get the total energy estimated per trip.
"""
energy_by_trip = routee_results.groupby("trip_id").agg(
    {"kilometers": "sum", "kWhs": "sum"}
)
energy_by_trip["miles"] = 0.6213712 * energy_by_trip["kilometers"]
energy_by_trip["kwh_per_mi"] = energy_by_trip["kWhs"] / energy_by_trip["miles"]
energy_by_trip.head(10)
energy_by_trip["kwh_per_mi"].describe()
"""
Note that the predicted energy consumption values are relatively low because the current RouteE Transit pipeline does not account for HVAC loads, which are a major contributor to BEB energy usage.
"""
"""

"""
