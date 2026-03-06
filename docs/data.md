# Data Requirements

(gtfs-reqs)=
## General Transit Feed Specification (GTFS)
RouteE-Transit relies on input data in the static [GTFS format](https://gtfs.org/) to generate energy predictions for each trip. Some features, such as complete shape traces for each trip, are optional per the GTFS standard but mandatory for RouteE-Transit analysis. RouteE-Transit requires the following GTFS tables and fields:

- *trips.txt*: `trip_id` and `shape_id` for all trips
- *shapes.txt*: complete shape traces (`shape_pt_lat`, `shape_pt_lon`, `shape_pt_sequence`) for each trip
- *stop_times.txt*: all stop time data for each `trip_id` included
- *stops.txt*: stop coordinates for any stop served on any trip to be analyzed


## Elevation Data
Road grade is computed automatically by RouteE-Compass using OSMnx to fetch elevation data from the USGS [National Map](https://www.usgs.gov/core-science-systems/national-geospatial-program/national-map). No manual download or configuration is required. For large geographic areas, the initial OSMnx network download may take additional time, but results are cached for subsequent runs.