### GTFS feed reading mechanisms for great glory

This package presents tooling for interacting with a GTFS Realtime feed as part of an effort to validate GTFS speed estimates (see [GitHub issue](https://github.com/NatLabRockies/routee-transit/issues/12)).

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
The original (clunky) process for processing live-scraped results to estimate and analyze speeds is:

1) Run `aggregate_agency_records.py` for a particular feed. Use `--single-day` for one JSONL file or `--data-dir` to process all days in a directory (skipping already-processed days). This can be very time consuming (multiple hours) for a large feed.
2) To plot the calculated speeds, use `build_speeds_map.py`

### Speed model training pipeline (current)

The bundled speed model was trained from the public [gtfsrt.io](https://gtfsrt.io)
archive, not from the live scraper above. Methodology, training data and
accuracy are documented in the package docs (`docs/data.md`, "Transit Speed
Model"). The pipeline is:

| Step | Script | Output |
|---|---|---|
| 1. Cache OSM networks per agency | `archive_speeds.py --stage networks` (`scripts/hpc/sbatch_fetch_networks.sh`) | osmnx cache under `--out-root` |
| 2. Estimate per-trip link speeds for 3 sampled weeks | `archive_speeds.py --stage analyze --sample-weeks 3` (`scripts/hpc/sbatch_run_speeds.sh`, one SLURM array task per agency) | `<out-root>/link_observations/agency=<slug>/date=<date>/` |
| 3. Train, evaluate and export the 12-bin model | `fit_poc_speed_model.py` (below) | ONNX model + manifest |
| 4. Ship it | copy `tuned_rf_speed_model.onnx` and `tuned_rf_speed_model_manifest.json` into `routee/transit/resources/speed_models/random-forest/` | bundled model |

`realtime_speeds.py` holds the shared map-matching, link-timing and dwell
detection code used by step 2. `export_speed_model_onnx.py` holds the ONNX
export/validation helpers used by step 3. To add an agency, add it to
`AGENCIES` in `archive_speeds.py` and the SLURM array in
`sbatch_run_speeds.sh`, and add its timezone to `AGENCY_TIMEZONES` in
`fit_poc_speed_model.py`.

`fit_speed_models.py` and `fit_archive_speed_models.py` trained the earlier
hourly model, which used scheduled speed as an input and ran ONNX inference
per edge inside Compass. That model format is no longer accepted by the
package; the scripts are kept because `fit_poc_speed_model.py` imports shared
constants from `fit_speed_models.py`.

Training with `--aggregated-csv` reproduces the bundled model's metrics to
about 0.001 mph, but predictions differ slightly because the CSV round-trip
loses some floating-point precision. Training from `--link-observations-root`
reproduces the bundled model's predictions exactly.

### 12-bin speed model proof of concept

From the repository root, using the development environment:

```bash
pixi run -e hpc python scripts/gtfs_realtime/fit_poc_speed_model.py \
    --link-observations-root gtfsrt_archive_runs/link_observations \
    --output-dir reports/speed_poc
```

This reads the processed archive without downloading data. One pooled random
forest uses six local-time periods (00-06, 06-09, 09-15, 15-19, 19-22,
22-24) crossed with weekday/weekend. The period is one-hot encoded; scheduled
speed and agency identity are not predictors. Naive UTC observation timestamps
are converted using the verified agency timezone map in the script. Adding an
agency requires adding its timezone; unknown agencies fail explicitly.

Speeds are observation-weighted within each road/period/day-type group, with
at least three contributing trip-link observations. The existing observed-only,
1-65 mph and per-road IQR filters are retained. Agency weights are normalized
so each agency has equal total influence. The forest has 100 trees, depth 8,
minimum leaf size 10, and uses four workers; there is no parameter search.

The only model comparison is time features versus no time features, holding out
whole agencies in three folds. Both models predict the same held-out groups,
with identical preprocessing and training weights. `time_comparison.csv`
reports observation-weighted MAE/RMSE per held-out agency; the console and model
manifest report equal-agency averages. Preprocessing medians are learned from
training agencies only. The final time model is refit on all agencies.

Outputs include the model bundle, ONNX model, feature manifest, aggregated
training data and comparison CSV. ONNX export is checked against sklearn.
Use the new model without replacing bundled resources:

```python
predictor = GTFSEnergyPredictor(
    gtfs_path="path/to/gtfs",
    speed_model_dir="reports/speed_poc",
    output_dir="reports/my_run",
    overwrite=False,
)
```

The bundled model uses this same 12-bin format. Older hourly model manifests
are rejected with a retraining instruction. Precomputed speeds are rounded to
0.1 mph before trace deduplication; cache format version 2 invalidates old
48-bin datasets.

#### Small fixed tuning trials

Reuse the same aggregated rows and held-out-agency folds without rereading
the archive:

```bash
pixi run -e hpc python scripts/gtfs_realtime/fit_poc_speed_model.py \
    --aggregated-csv reports/speed_poc/aggregated_training_data.csv \
    --trial all_features --output-dir reports/speed_poc_tuning/all_features
```

Available trials (all retain 100 trees and the original random seed):

| Trial | Depth | Minimum leaf size | Features per split | Stop density |
|---|---:|---:|---:|---|
| `current` | 8 | 10 | 50% | No |
| `all_features` | 8 | 10 | 100% | No |
| `deeper` | 12 | 20 | 100% | No |
| `stop_density` | 12 | 20 | 100% | Yes |
| `current_stop_density` | 8 | 10 | 50% | Yes |

Stop density is stops/km with a 50 m minimum denominator to avoid extreme
ratios on tiny links. Missing/nonpositive lengths remain missing for training-set
median imputation. The same calculation is shared with network preprocessing.
Each trial exports a separately usable model and its time/no-time comparison;
running a trial does not replace the bundled model.