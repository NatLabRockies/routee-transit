# Results QC

Quality checks for the output of a batch run (see [`../hpc/`](../hpc/)) before
the results are shared for T3CO runs or the FTA app. It only reads each
dataset's `results/` folder, so it runs on Kestrel or on a local copy of the
database, and doesn't need the RouteE-Transit build.

## Usage

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from
the repo root:

```bash
uv run --project scripts/results_qc python scripts/results_qc/qc_national_results.py \
    /path/to/routee-transit-db-10-21-2026 --out reports/qc_national
```

On Kestrel you can use the `hpc` env instead, which already has these packages:
`pixi run -e hpc python scripts/results_qc/qc_national_results.py <db_root>`.

`<db_root>` is the batch database folder: it holds `feeds.csv` and one folder per
dataset, each with a `results/` subfolder. Outputs go to `--out`
(`reports/` is gitignored):

| file | contents |
| --- | --- |
| `qc_report.html` | Self-contained report with all plots and tables |
| `agency_vehicle_summary.csv` | One row per dataset and bus: annual miles, deadhead and HVAC shares, energy per mile (native units and kWh-equivalent), trip-rate percentiles |
| `trip_rate_histograms.csv` | Trip-level kWh-equivalent per mile histograms per bus, weighted by days run |
| `daily_energy_by_agency.csv.gz` | Daily energy and block count per dataset and bus, from the block results |
| `dataset_status.csv` | Run status per dataset, joined to feed name, state and region |
| `flags.csv` | Results that fail a plausibility check |

## What it checks

- **Energy per mile** for every bus, at the agency and trip level, on one
  kWh-equivalent axis (33.705 kWh per GGE) so all fuels compare directly.
- **Region**: agency rates by Census region, and maps of agency rates.
- **Day to day**: energy is computed once per GTFS shape per bus, so only
  electric-bus HVAC changes from day to day. The report divides daily electric
  energy by same-size diesel energy to remove service-level changes and show the
  seasonal HVAC signal by region. It also shows service level by day of week.
- **Deadhead and HVAC shares** of miles and energy.
- **Flags**: rates outside loose screening ranges, deadhead over 25% of miles,
  HVAC over 40% of energy, over 1% of trips with zero miles or bad MPGe, and
  ordering checks within an agency (60 ft above 40 ft, hybrid below diesel,
  electric below diesel). The thresholds are constants at the top of the
  script. They are screening bounds, not validation targets.
