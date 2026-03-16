### GTFS feed reading mechanisms for great glory

This package presents tooling for interacting with a GTFS Realtime feed as part of an effort to validate GTFS speed estimates (see [GitHub issue](https://github.com/NREL/routee-transit/issues/12)).

### usage

  1. install uv [link](https://docs.astral.sh/uv/getting-started/installation/)
  2. create a venv: `cd gtfs_realtime; uv venv`
  3. install this package: `uv pip install .`
  4. run this package: `uv run scraper.py --log-dir denver https://www.rtd-denver.com/files/gtfs-rt/VehiclePosition.pb`

### configuration

to read about configuration values, run the --help command: `uv run scraper.py --help`

### results

scraped results are recorded in newline-delimited JSON format which can be loaded programmatically in pandas via:
```python
import pandas as pd
df = pd.read_json("denver/gtfs_realtime_records_20250811.jsonl", lines=True)
```

### feeds I'm currently scraping
King County, WA: `uv run scraper.py --interval 10 --log-dir kingcounty https://s3.amazonaws.com/kcm-alerts-realtime-prod/vehiclepositions.pb`
CDTA, Albany, NY: `uv run scraper.py --interval 10 --log-dir cdta_albany http://gtfs.cdta.org:8080/gtfsrealtime/VehiclePositions`
Portland, ME: `uv run scraper.py --interval 10 --log-dir greater_portland_me https://gtfsrt.gptd.cadavl.com/ProfilGtfsRt2_0RSProducer-GPTD/VehiclePosition.pb`


### Processing scraped data to estimate speeds
The current (clunky) process for processing results to estimate and analyze speeds is:

1) Run `aggregate_speeds.py` for a particular feed on a particular day. This can be very time consuming (multiple hours) for a large feed as of now.
2) To plot the calculated speeds, use `build_speeds_map.py`