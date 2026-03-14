# RouteE-Transit Prediction Pipeline

RouteE-Transit uses [RouteE-Compass](https://github.com/NatLabRockies/routee-compass) to predict the energy consumption of bus trips from a static GTFS feed. The pipeline converts GTFS data into road link-level features (speed, grade, and distance) that RouteE-Compass uses to estimate energy consumption for each trip.

The full workflow proceeds in four stages:

1. **Load and filter GTFS trips** — select the trips to model
2. **Infer deadhead trips** *(optional)* — add non-revenue travel between trips and to/from depots
3. **Prepare road link features** — map match shapes to the road network and extract speed and grade
4. **Predict energy and thermal impacts** — run RouteE-Compass models and optionally add HVAC loads

## 1) Load and Filter GTFS Trips

RouteE-Transit reads a standard GTFS feed directory and loads trips along with their shape traces, stop locations, and stop times. Users can optionally filter to a specific service date and/or a subset of routes. The shape traces and scheduled stop times are used downstream to estimate average speed and distance at the road link level.

See [](data:gtfs-reqs) for the full list of required GTFS files and fields.

## 2) Deadhead Trip Inference

Revenue service trips don't capture all of a bus's potential energy usage. Buses must also travel between the end of one trip and the start of the next (what we call *mid-block deadhead* and is often known as *interlining* when a bus switches between routes), and between the depot and the first or last stop of the day (*depot deadhead*). Deadheads are not included in the GTFS standard, but RouteE-Transit can infer and route both types automatically.

### Mid-block deadhead

For each block in the GTFS feed, consecutive revenue trips are examined. When the last stop of one trip does not coincide with the first stop of the next, a mid-block deadhead trip is created between those two points. Origin–destination pairs that are closer than 200 m receive a straight-line fallback geometry; all others are routed via RouteE-Compass.

### Depot deadhead (pull-out / pull-in)

Depot deadhead trips represent the pull-out (depot → first stop of the block) and pull-in (last stop of the block → depot) movements. The nearest depot for each block is selected by minimizing the combined pull-out and pull-in distance across all depot candidates. By default, depot locations are drawn from the [National Transit Database "Public Transit Facilities and Stations – 2023"](https://data.transportation.gov/stories/s/gd62-jzra) dataset. Custom depot locations can be supplied by passing a `depot_path` to `GTFSEnergyPredictor`.

All deadhead shapes are generated through RouteE-Compass using shortest-time routing on the OpenStreetMap road network. Unique origin–destination pairs are routed only once, so blocks that share identical endpoints incur no additional routing cost.

## 3) Road Link Feature Preparation

Shape traces (both revenue and deadhead) are upsampled to approximately 1 Hz resolution and then map-matched to OpenStreetMap road links using RouteE-Compass's LCSS (Longest Common Subsequence) map matcher. Each matched link is annotated with:

- **Distance** — derived from OSM road geometry
- **Grade** — elevation data from the USGS National Map, fetched automatically
- **Speed** — estimated from the scheduled time between GTFS stops and the cumulative shape distance traveled

## 4) Energy Prediction and Thermal Impacts

### Powertrain energy

RouteE-Compass — via a custom Rust extension bundled with RouteE-Transit — predicts energy consumption from the road link features computed above. Two transit bus models are included:

| Model name | Output unit |
|---|---|
| `Transit_Bus_Battery_Electric` | kWh |
| `Transit_Bus_Diesel` | gallons diesel |

In RouteE-Transit 0.3.0, both models simply apply a kinetic energy stop penalty at GTFS stop locations (modeled as 0.5mv²). Future release will refine the physical and thermal models used to account for the impacts of stops. Results are also expressed in miles-per-gallon equivalent (MPGe) using EPA/DOE GGE conversion factors for cross-fuel comparison.

### Thermal impacts (HVAC + BTMS)

For battery-electric buses, auxiliary loads from the HVAC system and battery thermal management system (BTMS) can represent a significant share of total energy consumption. When `add_hvac=True`, RouteE-Transit adds these loads using county-level Typical Meteorological Year (TMY3) weather data:

1. Each stop is spatially joined to its US Census county.
2. TMY3 files for the relevant counties are downloaded from the NREL Open Energy Data Initiative (OEDI) S3 bucket.
3. Hourly HVAC + BTMS power demand is looked up from a temperature-dependent table (derived from the literature) and integrated over each trip's scheduled time window.
4. Thermal energy is computed for three weather scenarios — **summer** (hottest day of year), **winter** (coldest day), and **median** — giving a range of thermal impact estimates per trip.

The resulting `hvac_energy_kWh` is added to the powertrain energy for electric models in the trip-level output.

# Using the GTFSEnergyPredictor Class

RouteE-Transit provides an object-oriented interface through the `GTFSEnergyPredictor` class:

```python
from routee.transit import GTFSEnergyPredictor

# Initialize predictor — vehicle_models and output_dir are set here
predictor = GTFSEnergyPredictor(
    gtfs_path="path/to/gtfs",
    vehicle_models=["Transit_Bus_Battery_Electric"],
    output_dir="reports/my_agency",  # optional; enables result caching
    # depot_path is optional - defaults to NTD depot locations
)

# Option 1: Use the convenience method (recommended)
trip_results = predictor.run(
    date="2023/08/02",
    routes=["205"],
    add_mid_block_deadhead=True,
    add_depot_deadhead=True,
    add_hvac=True,
)

# Option 2: Step-by-step processing for more control
predictor.load_gtfs_data()
predictor.filter_trips(date="2023/08/02", routes=["205"])
predictor.add_mid_block_deadhead()  # Between-trip deadhead
predictor.add_depot_deadhead()      # To/from depot (uses NTD locations)
predictor.get_link_level_inputs()   # Map matching + grade via RouteE-Compass
predictor.predict_energy(add_hvac=True)
```

# Assumptions and Limitations

- **Speed estimation**: Trip speed is assumed constant along each shape and is derived from scheduled stop times and cumulative shape distance. Actual in-service speed variation is not captured.
- **Deadhead speed**: Deadhead trips assume a uniform average speed of 30 km/h for travel time estimation.
- **Depot matching**: The nearest depot is chosen by minimising total pull-out + pull-in distance. Actual depot assignments may differ from operational practice.
- **TMY weather**: HVAC loads use typical (not actual) meteorological year data.
