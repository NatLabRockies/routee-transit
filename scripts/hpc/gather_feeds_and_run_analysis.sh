#!/bin/bash
#
#SBATCH --job-name=routee-ma
#SBATCH --account=teta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=104
#SBATCH --partition=medmem
#SBATCH --mem=960G
#SBATCH --time=48:00:00
#SBATCH --output=routee-transit-%j.out

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT" || exit 1

PIXI_ENV="${ROUTEE_PIXI_ENV:-hpc}"

# --- Configuration ---
MOBILITY_DB_DIR="/projects/teta/dmccabe/routee-transit-db-20260922"
FEEDS_CSV="$MOBILITY_DB_DIR/feeds.csv"
DATASETS_CSV="$MOBILITY_DB_DIR/datasets.csv"

# Phases to run. Override per submission so the Overpass-bound pre-cache gets
# its own wall-clock budget. Quote the whole --export value so the space in the
# phase list stays inside one argument:
#   jid=$(sbatch --parsable --export="ALL,ROUTEE_PHASES=feeds precache" ...)
#   sbatch --dependency=afterok:$jid --export="ALL,ROUTEE_PHASES=analyze" ...
PHASES="${ROUTEE_PHASES:-feeds precache analyze}"

# osmnx's Overpass response cache. Shared across runs on purpose: a bbox
# fetched once is never re-requested, which is the main defense against the
# public server refusing connections.
export ROUTEE_OSMNX_CACHE="${ROUTEE_OSMNX_CACHE:-$REPO_ROOT/cache}"

# overpass-api.de refuses TCP from NREL's egress NAT (192.174.62.50) on both
# :80 and :443 — the whole site shares ~2 Overpass slots, so it is effectively
# always blocked here. These mirrors serve the same worldwide data and were
# verified reachable from a compute node; re-verify with
# scripts/hpc/check_overpass_endpoints.py if downloads start failing again.
export ROUTEE_OVERPASS_URLS="${ROUTEE_OVERPASS_URLS:-https://overpass.kumi.systems/api,https://overpass.private.coffee/api,https://overpass.monicz.dev/api}"

# Datasets downloaded concurrently during pre-caching. The public Overpass
# server grants ~2 slots per IP and blocks the node IP under heavier sustained
# load, which is what failed the previous run.
PRECACHE_PARALLEL_DOWNLOADS=2

# Dataset analysis timeout in minutes (set to 0 to disable timeout)
DATASET_TIMEOUT_MINUTES=240

# Feed IDs to skip (space-separated list)
# mdb-2297: Amtrak Vermonter (includes one bus route)
# mdb-2321: Megabus
# mdb-419: Vineyard Fast Ferry, which includes a ferry route mistakenly labeled bus
# tld-5741: Coach USA
SKIP_FEED_IDS="mdb-2297 mdb-2321 mdb-419 tld-5741"

# Parallel processing configuration
MAX_PARALLEL_DATASETS=10
N_PROCESSES_PER_DATASET=8

# Run ID: use SLURM job ID if available, otherwise generate from timestamp
RUN_ID="${SLURM_JOB_ID:-local-$(date +%Y%m%d%H%M%S)}"
RUN_STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_phase() {
    [[ " $PHASES " == *" $1 "* ]]
}

extract_status_field() {
    local status_file="$1"
    local field_name="$2"
    [[ -f "$status_file" ]] || return 0
    sed -n "s/.*\"$field_name\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" "$status_file" | head -1
}

log "Run ID: $RUN_ID"
log "DB root: $MOBILITY_DB_DIR"
log "Repository: $REPO_ROOT"
log "Pixi environment: $PIXI_ENV"
log "Phases: $PHASES"
log "osmnx cache: $ROUTEE_OSMNX_CACHE"
log "Overpass endpoints: $ROUTEE_OVERPASS_URLS"

mkdir -p "$MOBILITY_DB_DIR"

# Maximum number of datasets to process (set to 0 to process all)
MAX_DATASETS=0

# --- Phase 1: Gather feeds ---
if run_phase feeds; then
    echo "========================================"
    log "Phase 1: Gathering all US feeds..."
    echo "========================================"
    pixi run -e "$PIXI_ENV" python scripts/feeds/gather_feeds.py \
        --db_root "$MOBILITY_DB_DIR"
    if [ $? -ne 0 ]; then
        log "ERROR: Feed gathering failed. Exiting."
        exit 1
    fi
else
    log "Skipping phase 1 (gather feeds)."
fi

if [ ! -f "$FEEDS_CSV" ] || [ ! -f "$DATASETS_CSV" ]; then
    log "ERROR: $FEEDS_CSV or $DATASETS_CSV is missing. Run the 'feeds' phase first."
    exit 1
fi

# --- Phase 1b: Pre-cache OSM networks ---
# Every dataset that reaches phase 2 without a cache downloads its own network,
# and ~10 of those at once is what got the node IP refused by Overpass.
if run_phase precache; then
    echo "========================================"
    log "Phase 1b: Pre-caching OSM networks..."
    echo "========================================"
    pixi run -e "$PIXI_ENV" python scripts/hpc/precache_osm_networks.py \
        "$MOBILITY_DB_DIR" \
        --parallel-downloads "$PRECACHE_PARALLEL_DOWNLOADS" \
        --n_processes 8 \
        --report "$MOBILITY_DB_DIR/precache_report.csv"
    if [ $? -ne 0 ]; then
        log "WARNING: Some OSM network downloads failed. See precache_report.csv."
    fi
else
    log "Skipping phase 1b (pre-cache OSM networks)."
fi

if ! run_phase analyze; then
    log "Skipping phase 2 (analysis) and phase 3 (aggregation)."
    exit 0
fi

# Phase 2 assumes every dataset already has a reusable cache. Anything missing
# would go back to Overpass under 10-way concurrency, so surface it up front.
log "Checking OSM cache coverage before analysis..."
pixi run -e "$PIXI_ENV" python scripts/hpc/precache_osm_networks.py \
    "$MOBILITY_DB_DIR" --check-only
if [ $? -ne 0 ]; then
    if [ "${ALLOW_UNCACHED:-0}" != "1" ]; then
        log "ERROR: Some datasets lack a reusable OSM cache. Rerun the 'precache'"
        log "       phase, or set ALLOW_UNCACHED=1 to analyze anyway."
        exit 1
    fi
    log "WARNING: Proceeding with incomplete cache coverage (ALLOW_UNCACHED=1)."
fi

# --- Phase 2: Collect qualifying datasets and run analysis ---
echo "========================================"
log "Phase 2: Scanning for qualifying datasets..."
echo "========================================"

declare -a QUALIFYING_DATASETS
while IFS=$'\t' read -r feed_id latest_dataset_id has_shapes has_errors includes_bus_trips includes_all_bus_shapes; do

    # Check if this feed ID should be skipped
    skip=0
    if [ -n "$SKIP_FEED_IDS" ]; then
        for skip_feed in $SKIP_FEED_IDS; do
            if [ "$feed_id" = "$skip_feed" ]; then
                log "Skipping dataset $latest_dataset_id: feed $feed_id is in skip list"
                skip=1
                break
            fi
        done
    fi
    [ $skip -eq 1 ] && continue

    # Check all conditions
    if [ "$has_shapes" != "True" ] && [ "$has_shapes" != "true" ] && [ "$has_shapes" != "1" ]; then
        continue
    fi
    if [ "$has_errors" == "True" ] || [ "$has_errors" == "true" ] || [ "$has_errors" == "1" ]; then
        continue
    fi
    if [ "$includes_bus_trips" != "True" ] && [ "$includes_bus_trips" != "true" ] && [ "$includes_bus_trips" != "1" ]; then
        continue
    fi
    if [ "$includes_all_bus_shapes" != "True" ] && [ "$includes_all_bus_shapes" != "true" ] && [ "$includes_all_bus_shapes" != "1" ]; then
        continue
    fi

    DATASET_DIR="$MOBILITY_DB_DIR/$latest_dataset_id"
    GTFS_DIR="$DATASET_DIR/gtfs"

    if [ ! -d "$DATASET_DIR" ]; then
        log "Warning: Dataset directory $DATASET_DIR does not exist. Skipping."
        continue
    fi
    if [ ! -d "$GTFS_DIR" ]; then
        log "Warning: GTFS directory $GTFS_DIR does not exist. Skipping."
        continue
    fi

    QUALIFYING_DATASETS+=("$latest_dataset_id")

    if [ $MAX_DATASETS -gt 0 ] && [ ${#QUALIFYING_DATASETS[@]} -ge $MAX_DATASETS ]; then
        log "Reached limit of $MAX_DATASETS qualifying datasets."
        break
    fi
done < <(pixi run -e "$PIXI_ENV" python scripts/hpc/parse_datasets_csv.py "$FEEDS_CSV" "$DATASETS_CSV")

log "Found ${#QUALIFYING_DATASETS[@]} qualifying datasets."
log "Running $MAX_PARALLEL_DATASETS datasets in parallel with $N_PROCESSES_PER_DATASET processes each."

echo "========================================"
log "Starting parallel batch processing"
if [ $DATASET_TIMEOUT_MINUTES -gt 0 ]; then
    log "Dataset timeout: $DATASET_TIMEOUT_MINUTES minutes per dataset"
else
    log "Dataset timeout: disabled"
fi
echo "========================================"

TIMEOUT_ARGS=""
if [ $DATASET_TIMEOUT_MINUTES -gt 0 ]; then
    TIMEOUT_ARGS="--timeout $DATASET_TIMEOUT_MINUTES"
fi

for latest_dataset_id in "${QUALIFYING_DATASETS[@]}"; do
    # Block until a worker slot is free
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL_DATASETS" ]; do
        wait -n 2>/dev/null || sleep 1
    done

    (
        RESULTS_DIR="$MOBILITY_DB_DIR/$latest_dataset_id/results"
        LOG_FILE="$RESULTS_DIR/${latest_dataset_id}.log"
        mkdir -p "$RESULTS_DIR"

        log "Starting: $latest_dataset_id"
        pixi run -e "$PIXI_ENV" python scripts/hpc/analyze_full_feed.py \
            --n_processes "$N_PROCESSES_PER_DATASET" \
            --run_id "$RUN_ID" \
            $TIMEOUT_ARGS \
            "$MOBILITY_DB_DIR" "$latest_dataset_id" > "$LOG_FILE" 2>&1
        exit_code=$?

        if [ $exit_code -eq 124 ]; then
            log "TIMEOUT: $latest_dataset_id exceeded $DATASET_TIMEOUT_MINUTES minutes. See $LOG_FILE"
            echo "$(date): TIMEOUT - Analysis exceeded $DATASET_TIMEOUT_MINUTES minutes limit" >> "$LOG_FILE"
        elif [ $exit_code -ne 0 ]; then
            STATUS_FILE="$RESULTS_DIR/status.json"
            failure_type=$(extract_status_field "$STATUS_FILE" "failure_type")
            error_summary=$(extract_status_field "$STATUS_FILE" "error_summary")

            if [[ "$failure_type" == overpass_* ]]; then
                log "OVERPASS: $latest_dataset_id ($failure_type). See $LOG_FILE"
                [[ -n "$error_summary" ]] && log "         Summary: $error_summary"
            elif [ $exit_code -eq 137 ]; then
                oom_msg=""
                if [[ -n "$SLURM_JOB_ID" ]]; then
                    oom_msg=$(dmesg -T 2>/dev/null | grep "oom-kill" | grep "job_${SLURM_JOB_ID}" | grep "task=python" | tail -1)
                fi
                log "OOM:     $latest_dataset_id (exit 137). See $LOG_FILE"
                if [[ -n "$oom_msg" ]]; then
                    echo "" >> "$LOG_FILE"
                    echo "$(date): ERROR - Process killed by Out-Of-Memory (OOM) killer" >> "$LOG_FILE"
                    echo "  $oom_msg" >> "$LOG_FILE"
                fi
            else
                log "FAILED:  $latest_dataset_id (exit code: $exit_code). See $LOG_FILE"
                [[ -n "$failure_type" ]] && log "         failure_type=$failure_type"
                [[ -n "$error_summary" ]] && log "         Summary: $error_summary"
            fi
        else
            log "Done:    $latest_dataset_id"
        fi
    ) &
done

# Wait for all remaining background jobs to finish
wait

echo "========================================"
log "All datasets processed."
echo "========================================"

# --- Phase 3: Aggregate results ---
log "Phase 3: Aggregating run status..."
pixi run -e "$PIXI_ENV" python scripts/hpc/aggregate_run_status.py \
    "$MOBILITY_DB_DIR" \
    --run_id "$RUN_ID" \
    --started_at "$RUN_STARTED_AT"

echo "========================================"
log "Pipeline complete."
echo "========================================"
