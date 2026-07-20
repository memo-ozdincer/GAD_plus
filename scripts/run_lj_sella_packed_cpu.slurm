#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/lj_sella_pack_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/lj_sella_pack_%A_%a.err
#SBATCH --array=0-27%28

# Packed CPU Sella sweep for future LJ runs.
#
# 4 Sella configs x 7 noise levels = 28 Slurm tasks.
# Each task owns one 64-core CPU allocation and runs the 8 sample partitions
# concurrently inside that allocation. This avoids hundreds of tiny Slurm jobs.

set -euo pipefail

module purge
module load StdEnv/2023 python/3.11

PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
SCRATCH="/lustre07/scratch/memoozd"
cd "$PROJECT_DIR"
source .venv/bin/activate

export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export TMPDIR="$SCRATCH/gadplus/tmp"
mkdir -p "$TMPDIR" "$SCRATCH/gadplus/logs"

N_SAMPLES="${N_SAMPLES:-287}"
PARTITIONS="${PARTITIONS:-8}"
N_STEPS_SELLA="${N_STEPS_SELLA:-1000}"
START_FROM="${START_FROM:-minimum_noised}"
ATOMIC_NUMBER="${ATOMIC_NUMBER:-1}"
FORCE_THRESHOLD="${FORCE_THRESHOLD:-0.01}"
OUT_ROOT="${OUT_ROOT:-$SCRATCH/gadplus/runs/lj_sella_packed_${SLURM_ARRAY_JOB_ID}}"

METHODS=(sella_carteck_d1 sella_carteck_d3 sella_cart_noeck_d1 sella_internal_d1)
NOISES=(0.00 0.01 0.03 0.05 0.10 0.15 0.20)

METHOD_IDX=$(( SLURM_ARRAY_TASK_ID / ${#NOISES[@]} ))
NOISE_IDX=$(( SLURM_ARRAY_TASK_ID % ${#NOISES[@]} ))
METHOD_NAME=${METHODS[$METHOD_IDX]}
NOISE=${NOISES[$NOISE_IDX]}

CHUNK=$(( (N_SAMPLES + PARTITIONS - 1) / PARTITIONS ))
OUT_DIR="$OUT_ROOT/$START_FROM/$METHOD_NAME/noise_${NOISE}"
mkdir -p "$OUT_DIR"

echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID method=$METHOD_NAME noise=$NOISE n=$N_SAMPLES out=$OUT_DIR"

run_partition() {
  local part_idx="$1"
  local sample_start=$(( part_idx * CHUNK ))
  local sample_end=$(( sample_start + CHUNK ))
  if (( sample_end > N_SAMPLES )); then
    sample_end=$N_SAMPLES
  fi
  if (( sample_start >= sample_end )); then
    return 0
  fi

  COMMON=(
    python -u scripts/lj_paper_runner.py
    --device cpu
    --method sella
    --n-atoms 7
    --n-samples "$N_SAMPLES"
    --sample-start "$sample_start"
    --sample-end "$sample_end"
    --noise "$NOISE"
    --start-from "$START_FROM"
    --atomic-number "$ATOMIC_NUMBER"
    --force-threshold "$FORCE_THRESHOLD"
    --n-steps "$N_STEPS_SELLA"
    --output-dir "$OUT_DIR"
  )

  case "$METHOD_NAME" in
    sella_carteck_d1)
      "${COMMON[@]}" --sella-cartesian --sella-apply-eckart --sella-diag-every 1 \
        --sella-delta0 0.048 --sella-gamma 0.0 --config-tag carteck_d1
      ;;
    sella_carteck_d3)
      "${COMMON[@]}" --sella-cartesian --sella-apply-eckart --sella-diag-every 3 \
        --sella-delta0 0.048 --sella-gamma 0.0 --config-tag carteck_d3
      ;;
    sella_cart_noeck_d1)
      "${COMMON[@]}" --sella-cartesian --no-sella-apply-eckart --sella-diag-every 1 \
        --sella-delta0 0.048 --sella-gamma 0.0 --config-tag cart_noeck_d1
      ;;
    sella_internal_d1)
      "${COMMON[@]}" --no-sella-apply-eckart --sella-diag-every 1 \
        --sella-delta0 0.048 --sella-gamma 0.0 --config-tag internal_d1
      ;;
    *)
      echo "Unknown METHOD_NAME=$METHOD_NAME" >&2
      return 2
      ;;
  esac
}

for part in $(seq 0 $((PARTITIONS - 1))); do
  run_partition "$part" &
done

wait
echo "done"
