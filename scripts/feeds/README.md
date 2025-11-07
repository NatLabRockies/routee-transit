# Gather Static GTFS Feeds from Mobility Database

## Overview

The `gather_feeds.py` script queries the [Mobility Database](https://mobilitydatabase.org/) to download and summarize GTFS Schedule data for transit systems in the United States. The script:

1. Queries active GTFS feeds from the Mobility Database
2. Downloads datasets that have shapes files and no validation errors
3. Extracts and verifies bus route/trip data
4. Generates summary CSV files with feed and dataset information

The datasets that are saved can then be subjected to the full RouteE-Transit energy prediction pipeline via separate procedures. Note that depending on the geographic scope that the script is run for, the amount of data downloaded could be quite large, since a single static GTFS dataset can often be a few hundred MB for a larger system.

## Prerequisites

Before running the script, you must obtain a Mobility Database refresh token from [https://mobilitydatabase.org/](https://mobilitydatabase.org/) and add it to your `.env` file:

```bash
MOBILITY_DATA_REFRESH_TOKEN=your_token_here
```

## Output

The script creates the following directory structure at `db_root`:

```
db_root/
├── feeds.csv           # Summary of all feeds (id, name, provider, location, etc.)
├── datasets.csv        # Summary of datasets (shapes, errors, bus trips, date ranges)
└── {dataset_id}/       # Individual dataset directories
    └── gtfs/           # Extracted GTFS files (routes.txt, trips.txt, shapes.txt, etc.)
```

## Running the script

### Gather feeds for a single state
```bash
python scripts/feeds/gather_feeds.py --db_root routee_transit_db --states Washington
```

### Gather feeds for multiple states
```bash
python scripts/feeds/gather_feeds.py --db_root routee_transit_db --states Washington Oregon California
```

### Gather feeds for all US states
```bash
python scripts/feeds/gather_feeds.py --db_root routee_transit_db
```

## Arguments

- `--db_root`: Root directory for storing GTFS datasets and feed info (default: `/reports/mdb`)
- `--states`: Optional list of state names (full names, space-separated). Omit to gather feeds from all US states.

## Dataset Filtering

The script only downloads datasets that meet the following criteria:
- Status: Active
- Country: United States
- Has shapes data
- No validation errors in the MobilityDB validation report
- Contains bus routes (route_type == 3)

## Output CSV Files

### feeds.csv
Contains one row per feed with columns:
- `id`: Feed ID from Mobility Database
- `name`: Feed name
- `provider`: Transit provider/agency name
- `status`: Feed status (active)
- `official`: Whether this is an official feed
- `latest_dataset_id`: ID of the most recent dataset
- `center_latitude`, `center_longitude`: Geographic center of the feed's bounding box

### datasets.csv
Contains one row per dataset with columns:
- `id`: Dataset ID
- `has_shapes`: Whether the dataset includes shapes.txt
- `has_errors`: Whether the dataset has validation errors
- `service_date_range_start`, `service_date_range_end`: Date range of service
- `hosted_url`: URL to download the dataset
- `includes_bus_trips`: Whether the dataset contains bus routes
- `includes_all_bus_shapes`: Whether all bus trips have associated shapes

## Visualizing Downloaded Feeds

After gathering feeds, you can create an interactive HTML map showing the geographic locations of all downloaded feeds:

```bash
python scripts/feeds/map_feeds.py routee_transit_db feeds_map.html
```

This will generate `feeds_map.html` in the current directory.