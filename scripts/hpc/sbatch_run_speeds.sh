#!/bin/bash
#
# Stage 2 of the GTFS-RT archive speed analysis: estimate per-link speeds for
# three full weeks (spread across each agency's archived date range) and write a
# model-ready parquet dataset. Runs as a SLURM job ARRAY — one task per agency,
# in parallel. This stage reuses the OSM networks cached by Stage 1
# (sbatch_fetch_networks.sh), so it makes NO Overpass calls.
#
# IMPORTANT: run Stage 1 to completion first (the array tasks would otherwise
# hit Overpass in parallel). Chain them with a dependency:
#     jid=$(sbatch --parsable scripts/hpc/sbatch_fetch_networks.sh)
#     sbatch --dependency=afterok:$jid scripts/hpc/sbatch_run_speeds.sh
#
# Re-run later for different dates by editing --sample-weeks / --dates below and
# resubmitting just this script — Stage 1 does not need to run again.
#
#SBATCH --job-name=rt-speeds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-10
#SBATCH --output=gtfsrt_archive_runs/logs/speeds-%A_%a.out
# Set your allocation if the cluster requires one:
##SBATCH --account=<your_account>
##SBATCH --partition=shared

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p gtfsrt_archive_runs/logs

# One array index per bus agency (must match archive_speeds.AGENCIES slugs).
AGENCIES=(
    dayton
    citybus-lafayette
    mountainline
    bigbluebus
    actransit
    broward
    cats
    gcrta
    metro-transit-mpls
    metro-transit-madison
    wrta-youngstown
)

AGENCY="${AGENCIES[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date '+%F %T')] Stage 2: analyzing $AGENCY (array task $SLURM_ARRAY_TASK_ID)"

# The hpc env must already be built (by Stage 1 / `pixi run -e hpc build_hpc`).
# Verify rather than rebuild here — parallel array tasks must not build the env
# concurrently (they would race on the shared install).
pixi run -e hpc python -c "import nrel.routee.compass, osmnx, pyarrow" || {
    echo "ERROR: hpc env not ready. Run Stage 1 (sbatch_fetch_networks.sh) or" >&2
    echo "       'pixi run -e hpc build_hpc' before submitting this array." >&2
    exit 1
}

# Three full weeks spread across the agency's archived range, model-ready parquet
# under gtfsrt_archive_runs/link_observations/agency=<slug>/date=<date>/part.parquet
pixi run -e hpc python scripts/gtfs_realtime/archive_speeds.py \
    --stage analyze \
    --agency "$AGENCY" \
    --sample-weeks 3 \
    --out-root gtfsrt_archive_runs

echo "[$(date '+%F %T')] Done: $AGENCY"
