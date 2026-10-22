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

echo "[$(date '+%F %T')] Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "[$(date '+%F %T')] CPUs: ${SLURM_CPUS_PER_TASK:-unset}; repo: ${REPO_ROOT}"

echo "[$(date '+%F %T')] Ensuring hpc env is built (build_hpc)…"
pixi run -e hpc build_hpc

echo "[$(date '+%F %T')] Running FTA app sample results"
pixi run -e hpc python scripts/fta_app_sample_results.py

echo "[$(date '+%F %T')] Done. Results are under reports/fta_results_101926"
