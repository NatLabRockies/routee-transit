#!/bin/bash
#
# Run the FTA app sample-results pipeline as a batch job.
#
# Submit from the repo root:
#   sbatch scripts/hpc/sbatch_fta_app_sample_results.sh
#
# Override resources at submission time when needed, e.g.:
#   sbatch --cpus-per-task=32 --mem=180G --time=24:00:00 \
#       scripts/hpc/sbatch_fta_app_sample_results.sh
# For a quick run under 4 hours, add --partition=short --time=4:00:00.
#
#SBATCH --job-name=fta-sample-results
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=fta-sample-results-%j.out
#SBATCH --account=teta
#SBATCH --partition=standard

set -euo pipefail

# SLURM runs a spooled copy of this script, so use the submit directory.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

# Geofabrik state extracts; unset or missing falls back to the Overpass API.
ROUTEE_OSM_PBF_DIR="${ROUTEE_OSM_PBF_DIR:-/projects/teta/osm/states}"
if [ -d "$ROUTEE_OSM_PBF_DIR" ]; then
    export ROUTEE_OSM_PBF_DIR
else
    unset ROUTEE_OSM_PBF_DIR
fi

# The site's XALT profiler preloads a library that corrupts the heap in the
# pyo3-ffi build script, so Compass cannot be built from source with it set.
unset LD_PRELOAD

echo "[$(date '+%F %T')] Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "[$(date '+%F %T')] CPUs: ${SLURM_CPUS_PER_TASK:-unset}; repo: ${REPO_ROOT}"
echo "[$(date '+%F %T')] OSM source: ${ROUTEE_OSM_PBF_DIR:-Overpass API}"

echo "[$(date '+%F %T')] Ensuring hpc env is built (build_hpc)…"
pixi run -e hpc build_hpc

echo "[$(date '+%F %T')] Running FTA app sample results"
pixi run -e hpc python scripts/fta_app_sample_results.py

echo "[$(date '+%F %T')] Done. Results are under reports/fta_results_101926"
