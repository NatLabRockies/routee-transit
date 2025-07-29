# RouteE-Transit Scripts
This directory contains scripts for running RouteE-Transit. Files included:

- `single_agency_full_analysis.py`: Script to run full OSM-based RouteE-Transit energy prediction pipeline for a single agency (by default, Utah Transit Authority in Salt Lake City). Note that to factor in road grade in energy predictions, USGS elevation raster files must be available (e.g., in `data/usgs_elevation`)