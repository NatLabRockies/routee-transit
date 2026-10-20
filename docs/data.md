# Data Requirements

(gtfs-reqs)=
## General Transit Feed Specification (GTFS)
RouteE-Transit relies on input data in the static [GTFS format](https://gtfs.org/) to generate energy predictions for each trip. Some features, such as complete shape traces for each trip, are optional per the GTFS standard but mandatory for RouteE-Transit analysis. In addition to all files and columns required by the GTFS standard, RouteE-Transit requires the following GTFS tables and fields:

- *trips.txt*: `trip_id` and `shape_id` for all trips
- *shapes.txt*: complete shape traces (`shape_pt_lat`, `shape_pt_lon`, `shape_pt_sequence`) for each trip
- *stop_times.txt*: all stop time data for each `trip_id` included
- *stops.txt*: stop coordinates for any stop served on any trip to be analyzed


## Elevation Data
Road grade is computed automatically by RouteE-Compass using OSMnx to fetch elevation data from the USGS [National Map](https://www.usgs.gov/core-science-systems/national-geospatial-program/national-map). No manual download or configuration is required. For large geographic areas, the initial OSMnx network download may take additional time, but results are cached for subsequent runs.

## Transit speed traces

During Compass dataset generation, RouteE-Transit runs the bundled ONNX speed
model over all road links in bounded batches. Each link has 12 bins: six local
time periods for weekdays, followed by the same six for weekends:
00:00-06:00, 06:00-09:00, 09:00-15:00, 15:00-19:00, 19:00-22:00,
and 22:00-24:00. Boundaries belong to the period starting at that time.
Predictions are floored at 1 mph; invalid model outputs stop preprocessing.

Speeds are rounded to 0.1 mph, then identical float32 traces share a trace ID. The cache
stores a dense little-endian uint32 edge-to-trace mapping in
`transit_speed_edge_traces.bin` and consecutive 12-value little-endian float32
mph traces in `transit_speed_traces.bin`. Storage is 4 bytes per edge plus
48 bytes per unique trace, excluding file metadata.

A single Compass application shares these immutable tables across queries.
Map matching, deadhead routing, elapsed travel time, and energy calculations
all use the same lookup speed (`edge_speed`, also exposed as `transit_speed`).
Existing output units are preserved: `edge_speed` is in km/h and
`transit_speed` is in mph.
The lookup selects the bin at entry to each edge using departure time plus
accumulated travel time, including turn delays. Hours and weekdays roll forward
across midnight, including GTFS departures beyond 24:00:00. Speeds are constant
within each period and are not interpolated within an edge. A* estimates
use the maximum trace speed (subject to a query's speed limit).

This does not expand energy queries across service dates: path calculation
remains once per matched shape and vehicle model. Trips sharing a shape use
its earliest departure; without a date filter, the representative service day
defaults to Monday. Calendar expansion and HVAC modeling happen afterward.
This preserves the existing representative-shape approximation, rather than
providing trip-specific departure-time energy estimates.

ONNX Runtime is needed for preprocessing, not during traversal. Cached datasets
are reused when an `output_dir` is provided and `overwrite=False`.
With cache reuse enabled, datasets
with the old per-query ONNX configuration, missing trace files, or changed
model/manifest files are rebuilt automatically. Adding vehicle models to an
otherwise valid cache preserves the traces without repeating inference.

The POC is one pooled random forest, not separate models by agency or period.
It uses road attributes, GTFS stop counts, six period indicators and a weekday
indicator. Scheduled speed is excluded because it is unavailable on arbitrary
network links. Training weights give each agency equal total influence.
See the [POC training instructions](../scripts/gtfs_realtime/README.md#12-bin-speed-model-proof-of-concept).