# Results
RouteE-Transit Energy predictions are stored within the `energy_predictions` attribute of `GTFSEnergyPredictor`. If `save_results=True` in `GTFSEnergyPredictor.run()`, they are written to CSV files as well. This includes the following files:

* `trip_energy_predictions.csv`: The main intended output of the RouteE-Transit pipeline, which includes energy predictions at the GTFS trip level, for each vehicle model and weather scenario analyzed.
* `link_energy_predictions.csv`: More granular energy predictions at the road link level. These can be used, for example, to map individual bus trips and see which segments are most energy intensive.

## `trip_energy_predictions.csv`
The RouteE-Transit results file (trip_energy_predictions.csv) includes the following columns:
* `trip_id`: GTFS trip ID
* `miles`: distance of trip in miles (calculated from GTFS shapes)
* `vehicle`: name of RouteE-Powertrain vehicle model used to generate energy prediction
* `energy_used`: estimated total energy consumed during this trip, including HVAC energy for electric vehicles
* `energy_unit`: unit of energy consumption for this trip
* `route_id`: GTFS route ID
* `service_id`: GTFS service ID
* `block_id`: GTFS block ID
* `shape_id`: GTFS shape ID
* `route_short_name`: GTFS short name of route served on this trip (the one typically used for displaying to riders, e.g., route number)
* `agency_id`: GTFS agency_id of the agency that operates this trip
* `trip_type`: the type of trip: `service` for passenger service trips defined in GTFS, `pull-out` for deadhead trips from the depot to first trip of the day, `pull-in` for deadhead trips from last trip of the day to depo, or `mid_block_deadhead` for deadhead connecting two service trips
* `start_time`: start time of the trip, merged in from GTFS *stop_times.txt*
* `end_time`: end time of the trip, merged in from GTFS *stop_times.txt*
* `trip_duration_minutes`: duration of the trip in minutes (based on start_time and end_time). not yet implemented for deadhead trips
* `trip_count`: the number of days in the service period described by this GTFS data that the trip operates (a positive integer)
* `from_trip`: for deadhead trips, the trip ID of the service trip before this deadhead trip
* `to_trip`: for deadhead trips, the trip ID of the service trip after this deadhead trip
* `scenario`: weather scenario used to estimate HVAC energy demand. Current options are “winter”, “summer”, or “median"
* `hvac_energy_kWh`: estimated HVAC energy consumption for electric buses, in kWh. Note this has already been added to energy_used in those cases.