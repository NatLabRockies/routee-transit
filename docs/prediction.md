# RouteE-Transit Prediction Pipeline

RouteE-Transit uses [RouteE-Compass](https://github.com/NatLabRockies/routee-compass) to predict the energy consumption of bus trips from a static GTFS feed. The pipeline converts GTFS data into road link-level features (speed, grade, and distance) that RouteE-Compass uses to estimate energy consumption for each trip.

The full workflow proceeds in four stages:

1. **Load and filter GTFS trips** — select the trips to model
2. **Infer deadhead trips** *(optional)* — add non-revenue travel between trips and to/from depots
3. **Prepare road link features** — map match shapes to the road network and extract speed and grade
4. **Predict energy and thermal impacts** — run RouteE-Compass models and optionally add HVAC loads

## 1) Load and Filter GTFS Trips

RouteE-Transit reads a standard GTFS feed directory and loads trips along with their shape traces, stop locations, and stop times. Users can optionally filter to a specific service date and/or a subset of routes. The shape traces determine the road links each trip traverses, and the scheduled stop times supply both the trip's time of day and its scheduled speed, which feed the link-level speed prediction described below.

### Route filtering

By default, route filtering is **trip-level**: only trips whose `route_short_name` appears in the `routes` list are included, regardless of what other routes share the same GTFS block. This gives the most intuitive results, especially for agencies where interlining is common (e.g., a bus block might serve both route 5 and route 21).

When deadhead trips are requested (`add_mid_block_deadhead=True` or `add_depot_deadhead=True`), filtering automatically switches to **block-level** mode. In block-level mode, entire blocks are excluded if *any* trip in the block belongs to a route not in the requested set. This is necessary because deadhead estimation requires complete blocks. If block-level filtering removes all trips but trip-level filtering would have kept some, the error message will explain this and suggest alternatives (either disabling deadhead or adding the additional interlined routes to the `routes` list).

When using `filter_trips()` directly, you can control this behavior with the `use_block_filter` parameter:

```python
predictor.load_gtfs_data()

# Trip-level filtering (default) — individual trips on route "5" are
# always kept, even if their block also serves route "21"
predictor.filter_trips(routes=["5"])

# Block-level filtering — only blocks that exclusively serve route "5"
predictor.filter_trips(routes=["5"], use_block_filter=True)
```

See [](data:gtfs-reqs) for the full list of required GTFS files and fields.

## 2) Deadhead Trip Inference

Revenue service trips don't capture all of a bus's potential energy usage. Buses must also travel between the end of one trip and the start of the next (what we call *mid-block deadhead* and is often known as *interlining* when a bus switches between routes), and between the depot and the first or last stop of the day (*depot deadhead*). Deadheads are not included in the GTFS standard, but RouteE-Transit can infer and route both types automatically.

### Mid-block deadhead

For each block in the GTFS feed, consecutive revenue trips are examined. When the last stop of one trip does not coincide with the first stop of the next, a mid-block deadhead trip is created between those two points. Origin–destination pairs that are closer than 200 m receive a straight-line fallback geometry; all others are routed via RouteE-Compass.

### Depot deadhead (pull-out / pull-in)

Depot deadhead trips represent the pull-out (depot → first stop of the block) and pull-in (last stop of the block → depot) movements. The nearest depot for each block is selected by minimizing the combined pull-out and pull-in distance across all depot candidates. Depot locations are drawn from the [National Transit Database 2024 Annual Facility Inventory](https://www.transit.dot.gov/ntd/data-product/2024-annual-database-facility-inventory) and the [National Transit Map Agencies](https://geodata.bts.gov/maps/ad6b0823f7364cac86c5421834eaba84) tables, both bundled with this package.

All deadhead shapes are generated through RouteE-Compass using shortest-time routing on the OpenStreetMap road network. Unique origin–destination pairs are routed only once, so blocks that share identical endpoints incur no additional routing cost.

## 3) Road Link Feature Preparation

Shape traces (both revenue and deadhead) are upsampled to approximately 1 Hz resolution and then map-matched to OpenStreetMap road links using RouteE-Compass's LCSS (Longest Common Subsequence) map matcher. Each matched link is annotated with:

- **Distance** — derived from OSM road geometry
- **Grade** — elevation data from the USGS National Map, fetched automatically
- **Speed** — predicted per link by a machine-learning transit speed model (see below)

### Transit speed prediction

Buses do not travel at the posted speed limit, so RouteE-Transit predicts a
transit-specific operating speed (`transit_speed`) for every matched link instead of
using the OSM-derived `edge_speed`. A random-forest model trained on observed
GTFS-realtime vehicle positions ships with the package and is used by default. Its
features are:

- static per-link attributes: posted speed, lane count, grade, link length, functional
  class (freeway / principal arterial / minor arterial / collector / local), and the
  number of GTFS stops on the link
- the GTFS scheduled speed for the trip
- time-of-day features (hour, weekday/weekend, peak period) resolved from each trip's
  scheduled departure time

The model is exported to ONNX and evaluated inside RouteE-Compass, so no Python-side
inference is needed during a run. A different model bundle — produced by
`scripts/gtfs_realtime/fit_speed_models.py` and
`scripts/gtfs_realtime/export_speed_model_onnx.py` — can be supplied through the
`speed_model_dir` argument of `GTFSEnergyPredictor`.

## 4) Energy Prediction and Thermal Impacts

### Powertrain energy

RouteE-Compass — via a custom Rust extension bundled with RouteE-Transit — predicts energy consumption from the road link features computed above. Eight transit bus models are included:

| Model name | Powertrain | Curb-mass estimate | Reported unit |
|---|---|---|---|
| `Transit_Bus_Electric_40ft_300kWh` | Battery electric, 300 kWh | 32,000 lb | kWh |
| `Transit_Bus_Electric_60ft_600kWh` | Battery electric, 600 kWh | 40,000 lb | kWh |
| `Transit_Bus_Diesel_40ft` | Diesel | 28,000 lb | gallons diesel |
| `Transit_Bus_Diesel_60ft` | Diesel | 40,000 lb | gallons diesel |
| `Transit_Bus_Hybrid_40ft` | Diesel hybrid | 32,000 lb | gallons diesel |
| `Transit_Bus_Hybrid_60ft` | Diesel hybrid | 45,000 lb | gallons diesel |
| `Transit_Bus_CNG_40ft` | Compressed natural gas | 30,000 lb | kWh |
| `Transit_Bus_CNG_60ft` | Compressed natural gas | 43,000 lb | kWh |

Each model is a RouteE-Powertrain model whose energy rate (kWh per kilometer) is a
function of link speed and road grade. The models are evaluated in Rust through a
binned interpolation grid over those two features, so per-link predictions are fast
enough to run across a whole feed. Combustion models predict in kWh internally and are
converted to their reported fuel unit before results are returned.

On top of the powertrain model, a kinetic-energy stop penalty (0.5mv², using the
model's mass estimate) is applied at GTFS stop locations to represent the
deceleration/re-acceleration cycle at each stop. Pass `include_stop_penalty=False` to
`run()` or `predict_energy()` to disable it.

Results are also expressed in miles-per-gallon equivalent (MPGe) in the `mpge` column,
using EPA/DOE GGE conversion factors for cross-fuel comparison.

### Thermal impacts (HVAC + BTMS)

For battery-electric buses, auxiliary loads from the HVAC system and battery thermal management system (BTMS) can represent a significant share of total energy consumption. When `add_hvac=True` (the default), RouteE-Transit adds these loads using county-level Typical Meteorological Year (TMY3) weather data:

1. Each stop is spatially joined to its US Census county.
2. TMY3 files for the relevant counties are downloaded from the NREL Open Energy Data Initiative (OEDI) S3 bucket and averaged into a single hourly temperature profile for the service area.
3. Hourly HVAC + BTMS power demand is looked up from a temperature-dependent table (derived from the literature) and integrated over each trip's scheduled time window.
4. Thermal energy is computed **per calendar day**, so a trip that operates on many dates gets one result row per date, each reflecting that day's typical weather.

Because the underlying weather source is a typical meteorological year, the `scenario`
column in the output is always `"TMY"`. Seasonal comparisons are made by grouping the
results on the `date` column (e.g. by month) rather than by scenario.

The number of dates modeled depends on how the run was configured:

- With a `date` filter, only that single service date is modeled.
- Without a `date` filter, the 365-day window containing the most service dates in the
  feed is modeled.
- With `scale_to_year=True`, the feed's typical weekday service patterns are also
  projected onto dates the feed does not cover, so the output spans a full year.
  Projected rows are flagged with `trip_is_within_gtfs_scope=False`.

The resulting `hvac_energy_kWh` is added to the powertrain energy for electric models in the trip-level output.

# Using the GTFSEnergyPredictor Class

RouteE-Transit provides an object-oriented interface through the `GTFSEnergyPredictor` class:

```python
from routee.transit import GTFSEnergyPredictor

# Initialize predictor — vehicle_models and output_dir are set here
predictor = GTFSEnergyPredictor(
    gtfs_path="path/to/gtfs",
    vehicle_models=["Transit_Bus_Electric_40ft_300kWh"],
    output_dir="reports/my_agency",  # optional; enables graph/result caching
)

# Option 1: Use the convenience method (recommended)
# By default, only revenue trips are included (no deadhead), and HVAC
# energy is added for electric models.
trip_results = predictor.run(
    date="2023/08/02",
    routes=["205"],
)

# Option 2: Include deadhead trips, and turn off the stop penalty
# When deadhead is enabled with route filtering, block-level filtering
# is used automatically to ensure complete blocks.
trip_results = predictor.run(
    date="2023/08/02",
    routes=["205"],
    add_mid_block_deadhead=True,
    add_depot_deadhead=True,
    include_stop_penalty=False,
)

# Option 3: Model a full year of service from a short feed
trip_results = predictor.run(
    routes=["205"],
    scale_to_year=True,
)
```

Deadhead inference has to be requested through `run()`; the routing steps it depends on
are not exposed as standalone public methods.

For finer control over the revenue-service-only workflow, the individual steps can be
called directly:

```python
predictor.load_gtfs_data()
predictor.filter_trips(date="2023/08/02", routes=["205"])
predictor.add_trip_times()          # start/end time and duration per trip
predictor.load_compass_app()        # builds/loads the OSM graph + energy models
predictor.predict_energy(add_hvac=True)   # map matching + energy prediction
predictor.save_results()

trips = predictor.get_trip_predictions()
links = predictor.get_link_predictions()
```

Note that `predict_energy()` performs map matching itself. `get_link_level_inputs()` is a
separate helper that produces link-level features (distance, travel time, geometry) for
inspection or export, and is not a prerequisite for `predict_energy()`.

# Assumptions and Limitations

- **Speed estimation**: Link speeds come from a general-purpose transit speed model
  trained on GTFS-realtime observations, not on the modeled agency's own operations.
  Agency- or corridor-specific congestion is not captured unless a custom model is
  supplied via `speed_model_dir`.
- **Deadhead speed**: Deadhead trips assume a uniform average speed of 30 km/h for travel time estimation.
- **Deadhead routing**: Deadhead paths are shortest-time routes on the OSM network;
  origin–destination pairs less than 200 m apart use a straight-line geometry instead.
- **Depot matching**: The nearest depot is chosen by minimising total pull-out + pull-in distance. Actual depot assignments may differ from operational practice.
- **TMY weather**: HVAC loads use typical (not actual) meteorological year data, and a
  single service-area-average temperature profile is applied to all trips.
- **Passenger load**: Vehicle mass is a fixed per-model estimate; ridership-dependent
  mass variation is not modeled.
