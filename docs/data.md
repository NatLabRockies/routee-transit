# Data Requirements

(gtfs-reqs)=
## General Transit Feed Specification (GTFS)
RouteE-Transit relies on input data in the static [GTFS format](https://gtfs.org/) to generate energy predictions for each trip. Some features, such as complete shape traces for each trip, are optional per the GTFS standard but mandatory for RouteE-Transit analysis. RouteE-Transit requires the following GTFS tables and fields:

- *trips.txt*: `trip_id` and `shape_id` for all trips
- *shapes.txt*: complete shape traces (`shape_pt_lat`, `shape_pt_lon`, `shape_pt_sequence`) for each trip
- *stop_times.txt*: all stop time data for each `trip_id` included
- *stops.txt*: stop coordinates for any stop served on any trip to be analyzed


## Elevation Tiles
RouteE-Transit uses [gradeit](https://github.com/NREL/gradeit) to add road grade to links. Gradeit requires local raster files of elevation data to be available for usage. RouteE-Transit will automatically download and cache these files from the USGS [National Map](https://www.usgs.gov/core-science-systems/national-geospatial-program/national-map). If your analysis covers a large geographic area, this could take several minutes the first time. Future runs relying on cached files will be much faster.