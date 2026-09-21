# Results
RouteE-Transit energy predictions are stored within the `energy_predictions` attribute of `GTFSEnergyPredictor` and are also returned by `run()`. If `save_results=True` in `GTFSEnergyPredictor.run()` (the default), they are written to files in `output_dir` as well:

* `trip_energy_predictions.csv`: The main intended output of the RouteE-Transit pipeline, which includes energy predictions at the GTFS trip level, for each vehicle model and each modeled date.
* `link_energy_predictions.csv`: More granular energy predictions at the road link level. These can be used, for example, to map individual bus trips and see which segments are most energy intensive.
* `tods/`: [TODS](https://tods-transit.org/spec/) supplement files describing the inferred deadhead trips, written only when deadhead trips were added.

In code, the same results are reachable through:

```python
predictor.get_trip_predictions()                                    # all models
predictor.get_trip_predictions("Transit_Bus_Electric_40ft_300kWh")  # one model
predictor.get_link_predictions()
```

## `trip_energy_predictions.csv`
The trip-level results file includes the following columns:

### Energy prediction
* `energy_used`: estimated total energy consumed during this trip, including HVAC energy for electric vehicles
* `energy_unit`: unit of `energy_used` for this vehicle model (e.g., `kWh`, `gallons_diesel`)
* `vehicle`: name of the vehicle model used to generate the energy prediction (e.g., `Transit_Bus_Electric_40ft_300kWh`)
* `miles`: distance of trip in miles (from the map-matched road links)
* `mpge`: efficiency in miles per gallon of gasoline equivalent, allowing comparison across powertrains

### Weather and date
These columns are present only when HVAC energy was added (`add_hvac=True`, the default).
A trip that operates on multiple dates produces one row per date, per vehicle model.

* `date`: the calendar date this row models
* `scenario`: weather scenario used to estimate HVAC energy demand. Always `TMY` (Typical Meteorological Year)
* `hvac_energy_kWh`: estimated HVAC + BTMS energy consumption for electric buses, in kWh. Note this has already been added to `energy_used`
* `trip_is_within_gtfs_scope`: `True` when the date comes from the feed's own calendar, `False` for dates synthesized by `scale_to_year=True`

### Trip and schedule identification
* `trip_id`: GTFS trip ID
* `route_id`: GTFS route ID
* `route_short_name`: GTFS short name of route served on this trip (the one typically used for displaying to riders, e.g., route number)
* `route_color`: GTFS route color, when provided by the feed
* `block_id`: GTFS block ID
* `shape_id`: GTFS shape ID
* `agency_id`: GTFS agency_id of the agency that operates this trip
* `trip_type`: the type of trip: `service` for passenger service trips defined in GTFS, `pull-out` for deadhead trips from the depot to the first trip of the day, `pull-in` for deadhead trips from the last trip of the day to the depot, or `mid_block_deadhead` for deadhead connecting two service trips
* `start_time`: start time of the trip, merged in from GTFS *stop_times.txt*
* `end_time`: end time of the trip, merged in from GTFS *stop_times.txt*
* `trip_duration_minutes`: duration of the trip in minutes (based on start_time and end_time)
* `from_trip`: for deadhead trips, the trip ID of the service trip before this deadhead trip
* `to_trip`: for deadhead trips, the trip ID of the service trip after this deadhead trip
* `feed_id` / `dataset_id`: pass-through identifiers, present only when supplied to `GTFSEnergyPredictor`

The exact column set depends on the feed and the run configuration: optional GTFS fields
are carried through when present, and columns such as `date` or `from_trip` are absent
when HVAC or deadhead estimation is not used.

## `link_energy_predictions.csv`
The link-level file contains one row per traversed road link, per shape, per vehicle model.
Each row combines the map-matching result with the per-link state reported by
RouteE-Compass, including:

* `shape_id`, `edge_id`, `edge_list_id`, `edge_index`, `match_id`: identifiers tying the link back to the GTFS shape and the RouteE-Compass road network
* `edge_distance`, `edge_speed`, `edge_grade`, `edge_time`, `edge_turn_delay`: per-link road and traversal attributes
* `edge_energy_electric` / `edge_energy_liquid`: per-link energy in kWh, as reported by the powertrain model
* `trip_distance`, `trip_time`, `trip_elevation_gain`, `trip_elevation_loss`, `trip_soc`: cumulative values along the shape up to and including this link
* `vehicle`, `energy_used`, `energy_unit`: the vehicle model and the shape-level totals, repeated on every link for convenience
* `geometry`: the link geometry

Note that link-level results are keyed by `shape_id`, not `trip_id`: trips that share a
shape share the same link-level prediction.
