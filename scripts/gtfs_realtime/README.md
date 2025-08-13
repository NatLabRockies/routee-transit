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