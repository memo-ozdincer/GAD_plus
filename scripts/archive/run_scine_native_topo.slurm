#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/scine_native_topo_%A.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/scine_native_topo_%A.err

# Calculator-native DFTB0 IRC validation for one method/noise cell.
#
# Required environment variables:
#   SUMMARY_PARQUET, NOISE_PM, METHOD_TAG, OUTPUT_DIR, NATIVE_CACHE_DIR
#
# Optional:
#   TRAJ_DIR, SAMPLE_IDS, MAX_VALIDATE, N_WORKERS, MAX_IRC_STEPS,
#   MAX_RELAX_STEPS, RELAX_FMAX, RMSD_THRESHOLD, COLLAPSE_RMSD, H5, SPLIT

set -euo pipefail

module purge
module load StdEnv/2023 python/3.11

PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate

export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

: "${SUMMARY_PARQUET:?SUMMARY_PARQUET is required}"
: "${NOISE_PM:?NOISE_PM is required}"
: "${METHOD_TAG:?METHOD_TAG is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${NATIVE_CACHE_DIR:?NATIVE_CACHE_DIR is required}"

N_WORKERS="${N_WORKERS:-$SLURM_CPUS_PER_TASK}"
MAX_IRC_STEPS="${MAX_IRC_STEPS:-500}"
MAX_RELAX_STEPS="${MAX_RELAX_STEPS:-500}"
RELAX_FMAX="${RELAX_FMAX:-0.001}"
RMSD_THRESHOLD="${RMSD_THRESHOLD:-0.3}"
COLLAPSE_RMSD="${COLLAPSE_RMSD:-0.05}"
H5="${H5:-/lustre06/project/6033559/memoozd/data/transition1x.h5}"
SPLIT="${SPLIT:-test}"

mkdir -p "$OUTPUT_DIR" "$NATIVE_CACHE_DIR" /lustre07/scratch/memoozd/gadplus/logs

args=(
  --summary-parquet "$SUMMARY_PARQUET"
  --noise-pm "$NOISE_PM"
  --method-tag "$METHOD_TAG"
  --output-dir "$OUTPUT_DIR"
  --native-cache-dir "$NATIVE_CACHE_DIR"
  --n-workers "$N_WORKERS"
  --max-irc-steps "$MAX_IRC_STEPS"
  --max-relax-steps "$MAX_RELAX_STEPS"
  --relax-fmax "$RELAX_FMAX"
  --rmsd-threshold "$RMSD_THRESHOLD"
  --collapse-rmsd "$COLLAPSE_RMSD"
  --h5 "$H5"
  --split "$SPLIT"
)

if [[ -n "${TRAJ_DIR:-}" ]]; then
  args+=(--traj-dir "$TRAJ_DIR")
fi
if [[ -n "${SAMPLE_IDS:-}" ]]; then
  args+=(--sample-ids "$SAMPLE_IDS")
fi
if [[ -n "${MAX_VALIDATE:-}" ]]; then
  args+=(--max-validate "$MAX_VALIDATE")
fi

echo "method=$METHOD_TAG noise=$NOISE_PM workers=$N_WORKERS samples=${SAMPLE_IDS:-all}"
echo "summary=$SUMMARY_PARQUET"
echo "cache=$NATIVE_CACHE_DIR"

python -u scripts/scine_native_topo_validate.py "${args[@]}"
