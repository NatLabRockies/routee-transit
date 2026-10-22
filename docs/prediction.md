# RouteE-Transit Prediction Pipeline

RouteE-Transit uses [RouteE-Compass](https://github.com/NatLabRockies/routee-compass) to predict the energy consumption of bus trips from a static GTFS feed. The pipeline converts GTFS data into road link-level features (speed, grade, and distance) that RouteE-Compass uses to estimate energy consumption for each trip.

The full workflow proceeds in four stages:

1. **Load and filter GTFS trips** — select the trips to model
2. **Infer deadhead trips** *(optional)* — add non-revenue travel between trips and to/from depots
3. **Prepare road link features** — map match shapes to the road network and extract speed and grade
4. **Predict energy and thermal impacts** — run RouteE-Compass models and optionally add HVAC loads

## 1) Load and Filter GTFS Trips

RouteE-Transit reads a standard GTFS feed directory and loads trips along with their shape traces, stop locations, and stop times. Users can optionally filter to a specific service date and/or a subset of routes. The shape traces and scheduled stop times are used downstream to estimate average speed and distance at the road link level.

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
- **Speed** — estimated from the scheduled time between GTFS stops and the cumulative shape distance traveled

## 4) Energy Prediction and Thermal Impacts

### Powertrain energy

RouteE-Compass — via a custom Rust extension bundled with RouteE-Transit — predicts energy consumption from the road link features computed above. Two transit bus models are included:

| Model name | Output unit |
|---|---|
| `Transit_Bus_Electric_40ft_300kWh` | kWh |
| `Transit_Bus_Diesel_40ft` | gallons diesel |

In RouteE-Transit 0.3.0, both models simply apply a kinetic energy stop penalty at GTFS stop locations (modeled as 0.5mv²). Future release will refine the physical and thermal models used to account for the impacts of stops. Results are also expressed in miles-per-gallon equivalent (MPGe) using EPA/DOE GGE conversion factors for cross-fuel comparison.

### Thermal impacts (HVAC + BTMS)

For battery-electric buses, auxiliary loads from the HVAC system and battery thermal management system (BTMS) can represent a significant share of total energy consumption. When `add_hvac=True`, RouteE-Transit adds these loads using county-level Typical Meteorological Year (TMY3) weather data:

1. Each stop is spatially joined to its US Census county.
2. TMY3 files for the relevant counties are downloaded from the NREL Open Energy Data Initiative (OEDI) S3 bucket.
3. Hourly HVAC + BTMS power demand is looked up from a temperature-dependent table (derived from the literature) and integrated over each trip's scheduled time window.
4. Thermal energy is computed for each modeled service date using its TMY temperature profile. These rows use the **TMY** scenario.

The resulting `hvac_energy_kWh` is added to the powertrain energy for electric models in the trip-level output.

### Daily trip and block results

All vehicle models produce one trip row per modeled service date, including diesel models and runs with `add_hvac=False`. Dates come from the feed's service calendar, including calendar exceptions, using the densest 365-day window by default. A `date` filter restricts the output to that service date. Trips with times beyond 24:00 remain assigned to their original service date.

With `scale_to_year=True` and no date filter, typical weekday service is projected onto uncovered dates in the selected year window. Trip output includes `trip_is_within_gtfs_scope`: false identifies projected service. This calendar expansion applies independently of HVAC. Compared with earlier versions, non-electric and HVAC-disabled trip exports now contain repeated trips on different dates rather than one undated row per trip.

```python
trip_results = predictor.run(
    add_mid_block_deadhead=True,
    add_depot_deadhead=True,
)
block_results = predictor.get_block_predictions()
electric_blocks = predictor.get_block_predictions("Transit_Bus_Electric_40ft_300kWh")
```

Block results have one row per `vehicle`, `date`, and usable `block_id`:

| Column | Meaning |
|---|---|
| `energy_used_without_deadhead` | Sum of modeled revenue service trip energy, including HVAC when enabled |
| `energy_used_with_deadhead` | Sum of all modeled trip energy, including pull-out, pull-in, and mid-block deadhead |
| `energy_unit` | The vehicle model's output unit; incompatible units cannot be combined |
| `route_ids` | Sorted unique revenue route IDs, excluding synthetic deadhead routes |
| `agency_ids` | Sorted unique revenue agency IDs; a block interlined across agencies lists all of them |
| `feasible_without_deadhead` | Whether revenue energy fits the vehicle's usable capacity; unset for models without declared capacity |
| `feasible_with_deadhead` | Whether total energy including deadhead fits the vehicle's usable capacity |

#### One row per route

Consumers that cannot store list-valued columns — relational databases in particular — can request an exploded block schema with `explode_block_routes=True` on the constructor:

```python
predictor = GTFSEnergyPredictor(
    gtfs_path="path/to/gtfs",
    vehicle_models=["Transit_Bus_Electric_40ft_300kWh"],
    explode_block_routes=True,
)
```

The `route_ids` and `agency_ids` lists are then replaced by scalar `route_id` and `agency_id` columns, and each block contributes one row per revenue route it covers, repeating the block's energy totals and feasibility flags on every row. Agency comes from `routes.txt`, so each route contributes exactly one pairing. Summing an energy column across the exploded rows therefore double counts interlined blocks; aggregate over distinct `vehicle`, `date`, and `block_id` first. Blocks whose revenue trips carry no route ID keep a single row with null identifiers. The default rolled-up schema is unchanged.

Feasibility compares each total against `capacity * usable_capacity_fraction` from the vehicle model config. `save_results()` exports those capacities to `vehicles.csv`; pass `save_vehicle_specs=False` to skip it.

Both energy columns are equal when no deadhead is modeled. Accessing block results does not enable deadhead routing: the total includes only deadhead requested and successfully predicted by the pipeline. Results describe modeled trips, so route filtering or failed map matching can leave a partial block. A block left with only deadhead trips is dropped entirely, with a single warning counting the affected blocks, since its revenue service was excluded upstream and the remaining total is not a meaningful block.

Absent, null, empty, and whitespace-only block IDs are excluded from block aggregation, with a warning counting omitted trips. They remain in trip results. Deadhead inference skips these trips; with route filtering enabled, they are filtered individually while assigned blocks retain exclusive block filtering. If no usable IDs exist, block results are empty, not an error.

`save_results()` writes `block_energy_predictions.csv` beside `trip_energy_predictions.csv` and `link_energy_predictions.csv`, including a header-only block file when no blocks qualify. Route and agency lists are JSON arrays in the CSV, or plain values under `explode_block_routes=True`; dates use `YYYY-MM-DD`. Feed and dataset metadata are included in the trip export when supplied. Each export can be skipped independently with `save_trip_level=False`, `save_link_level=False`, or `save_block_level=False`. The existing trip filename and the trip DataFrame returned by `run()` are unchanged.

`save_results()` also writes `routes.geojson`, one EPSG:4326 feature per revenue route drawn on the shape its trips use most often. Each feature carries the route's identifiers, whichever of `route_short_name`, `agency_id`, and `route_color` the feed supplies, and the `mean_energy_used` and `mean_efficiency` objects keyed by vehicle model, each with a `value` and `unit`. Efficiency is reported as `kWh/mi` for models whose energy is in kWh and `mpg_diesel` otherwise, and mean energy is per trip day. Synthetic deadhead routes are excluded. Pass `save_route_shapes=False` to skip it.

# Using the GTFSEnergyPredictor Class

RouteE-Transit provides an object-oriented interface through the `GTFSEnergyPredictor` class:

```python
from routee.transit import GTFSEnergyPredictor

# Initialize predictor — vehicle_models and output_dir are set here
predictor = GTFSEnergyPredictor(
    gtfs_path="path/to/gtfs",
    vehicle_models=["Transit_Bus_Electric_40ft_300kWh"],
    output_dir="reports/my_agency",  # optional; enables result caching
)

# Option 1: Use the convenience method (recommended)
# By default, only revenue trips are included (no deadhead).
trip_results = predictor.run(
    date="2023/08/02",
    routes=["205"],
)

# Option 2: Include deadhead trips and HVAC impacts
# When deadhead is enabled with route filtering, block-level filtering
# is used automatically to ensure complete blocks.
trip_results = predictor.run(
    date="2023/08/02",
    routes=["205"],
    add_mid_block_deadhead=True,
    add_depot_deadhead=True,
    add_hvac=True,
)

# Option 3: Step-by-step processing for more control
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
