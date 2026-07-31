#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/ljcpu_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/ljcpu_%A_%a.err
#SBATCH --array=0-783%200

# CPU-only LJ analogue of the paper matrix.
#
# 14 method/configs x 7 noise levels x 8 sample partitions = 784 tasks.
# This intentionally avoids GPU/MIG allocations; LJ7 Hessians are tiny and
# the workload is mostly Python/Sella/Torch overhead plus small dense linear algebra.

set -euo pipefail

module purge
module load StdEnv/2023 python/3.11

PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
SCRATCH="/lustre07/scratch/memoozd"
cd "$PROJECT_DIR"
source .venv/bin/activate

export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export WANDB_DISABLED=true
export WANDB_MODE=disabled
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
START_FROM="${START_FROM:-minimum_noised}"
ATOMIC_NUMBER="${ATOMIC_NUMBER:-1}"
FORCE_THRESHOLD="${FORCE_THRESHOLD:-0.01}"
N_STEPS_GAD="${N_STEPS_GAD:-1000}"
N_STEPS_SELLA="${N_STEPS_SELLA:-1000}"
OUT_ROOT="${OUT_ROOT:-$SCRATCH/gadplus/runs/lj_paper_cpu_${SLURM_ARRAY_JOB_ID}}"

METHODS=(
  gad_dt0005
  gad_dt001
  gad_dt002
  gad_dt003
  gad_dt005
  gad_dt007
  gad_dt010
  gad_dt020
  hybrid_eckart_dt010_sf10_tr005
  hybrid_damped_eckart_dt010_sf10_tr005
  sella_carteck_d1
  sella_carteck_d3
  sella_cart_noeck_d1
  sella_internal_d1
)
NOISES=(0.00 0.01 0.03 0.05 0.10 0.15 0.20)

N_METHODS=${#METHODS[@]}
N_NOISE=${#NOISES[@]}
TASK=${SLURM_ARRAY_TASK_ID}
PART_IDX=$(( TASK % PARTITIONS ))
NOISE_IDX=$(( (TASK / PARTITIONS) % N_NOISE ))
METHOD_IDX=$(( TASK / (PARTITIONS * N_NOISE) ))

if (( METHOD_IDX >= N_METHODS )); then
  echo "Task $TASK maps outside method list; exiting."
  exit 0
fi

METHOD_NAME=${METHODS[$METHOD_IDX]}
NOISE=${NOISES[$NOISE_IDX]}

CHUNK=$(( (N_SAMPLES + PARTITIONS - 1) / PARTITIONS ))
SAMPLE_START=$(( PART_IDX * CHUNK ))
SAMPLE_END=$(( SAMPLE_START + CHUNK ))
if (( SAMPLE_END > N_SAMPLES )); then
  SAMPLE_END=$N_SAMPLES
fi

OUT_DIR="$OUT_ROOT/$START_FROM/$METHOD_NAME/noise_${NOISE}"
mkdir -p "$OUT_DIR"

echo "job=$SLURM_JOB_ID task=$TASK method=$METHOD_NAME noise=$NOISE partition=$PART_IDX samples=$SAMPLE_START:$SAMPLE_END out=$OUT_DIR"

COMMON=(
  python -u scripts/lj_paper_runner.py
  --device cpu
  --n-atoms 7
  --n-samples "$N_SAMPLES"
  --sample-start "$SAMPLE_START"
  --sample-end "$SAMPLE_END"
  --noise "$NOISE"
  --start-from "$START_FROM"
  --atomic-number "$ATOMIC_NUMBER"
  --force-threshold "$FORCE_THRESHOLD"
  --output-dir "$OUT_DIR"
)

case "$METHOD_NAME" in
  gad_dt0005)
    "${COMMON[@]}" --method gad --dt 0.0005 --n-steps "$N_STEPS_GAD"
    ;;
  gad_dt001)
    "${COMMON[@]}" --method gad --dt 0.001 --n-steps "$N_STEPS_GAD"
    ;;
  gad_dt002)
    "${COMMON[@]}" --method gad --dt 0.002 --n-steps "$N_STEPS_GAD"
    ;;
  gad_dt003)
    "${COMMON[@]}" --method gad --dt 0.003 --n-steps "$N_STEPS_GAD"
    ;;
  gad_dt005)
    "${COMMON[@]}" --method gad --dt 0.005 --n-steps "$N_STEPS_GAD"
    ;;
  gad_dt007)
    "${COMMON[@]}" --method gad --dt 0.007 --n-steps "$N_STEPS_GAD"
    ;;
  gad_dt010)
    "${COMMON[@]}" --method gad --dt 0.010 --n-steps "$N_STEPS_GAD"
    ;;
  gad_dt020)
    "${COMMON[@]}" --method gad --dt 0.020 --n-steps "$N_STEPS_GAD"
    ;;
  hybrid_eckart_dt010_sf10_tr005)
    "${COMMON[@]}" --method hybrid_eckart --n-steps "$N_STEPS_SELLA" \
      --gad-dt 0.010 --switch-force 10.0 --trust-radius 0.05
    ;;
  hybrid_damped_eckart_dt010_sf10_tr005)
    "${COMMON[@]}" --method hybrid_damped_eckart --n-steps "$N_STEPS_SELLA" \
      --gad-dt 0.010 --switch-force 10.0 --trust-radius 0.05 \
      --target-mode-strategy neg_force_coupling --high-index-descent index_controlled
    ;;
  sella_carteck_d1)
    "${COMMON[@]}" --method sella --n-steps "$N_STEPS_SELLA" \
      --sella-cartesian --sella-apply-eckart --sella-diag-every 1 \
      --sella-delta0 0.048 --sella-gamma 0.0 --config-tag carteck_d1
    ;;
  sella_carteck_d3)
    "${COMMON[@]}" --method sella --n-steps "$N_STEPS_SELLA" \
      --sella-cartesian --sella-apply-eckart --sella-diag-every 3 \
      --sella-delta0 0.048 --sella-gamma 0.0 --config-tag carteck_d3
    ;;
  sella_cart_noeck_d1)
    "${COMMON[@]}" --method sella --n-steps "$N_STEPS_SELLA" \
      --sella-cartesian --no-sella-apply-eckart --sella-diag-every 1 \
      --sella-delta0 0.048 --sella-gamma 0.0 --config-tag cart_noeck_d1
    ;;
  sella_internal_d1)
    "${COMMON[@]}" --method sella --n-steps "$N_STEPS_SELLA" \
      --no-sella-apply-eckart --sella-diag-every 1 --sella-delta0 0.048 \
      --sella-gamma 0.0 --config-tag internal_d1
    ;;
  *)
    echo "Unknown METHOD_NAME=$METHOD_NAME"
    exit 2
    ;;
esac

echo "done"
