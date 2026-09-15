#!/bin/bash
#
# Stage 1 of the GTFS-RT archive speed analysis: prefetch and cache each bus
# agency's OSM road network. This is the Overpass-heavy step and is run ONCE,
# sequentially, to stay gentle on overpass-api.de. Stage 2 (the analysis) then
# reuses these cached networks and never touches Overpass.
#
# Submit:  sbatch scripts/hpc/sbatch_fetch_networks.sh
# Then, after it finishes, submit the analysis array:
#          sbatch --dependency=afterok:<this_job_id> scripts/hpc/sbatch_run_speeds.sh
#
#SBATCH --job-name=rt-fetch-networks
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=gtfsrt_archive_runs/logs/fetch-networks-%j.out
# Set your allocation if the cluster requires one:
##SBATCH --account=<your_account>
##SBATCH --partition=shared

set -euo pipefail

# Run from the repo root so the ./cache (elevation) and out-root are consistent
# between this stage and the analysis stage.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p gtfsrt_archive_runs/logs

# Ensure the hpc env is built from source (ONNX Runtime + compass) before use.
# The PyPI compass wheels are mis-tagged for a newer glibc than this host has, so
# build_hpc compiles compass against a locally-built ORT. The task is cached, so
# this is a near no-op once built. `pixi run -e hpc` does not clobber the source
# build (it only re-syncs the lock, which keeps the same compass version).
echo "[$(date '+%F %T')] Ensuring hpc env is built (build_hpc)…"
pixi run -e hpc build_hpc

echo "[$(date '+%F %T')] Stage 1: prefetching OSM networks for all bus agencies"
pixi run -e hpc python scripts/gtfs_realtime/archive_speeds.py \
    --stage networks \
    --agency all \
    --out-root gtfsrt_archive_runs

echo "[$(date '+%F %T')] Done. Networks cached under gtfsrt_archive_runs/.osmnx_cache"
