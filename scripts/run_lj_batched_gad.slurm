#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/ljbatched_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/ljbatched_%A_%a.err
#SBATCH --array=0-55%20

# Batched GAD-only LJ7 dt/noise sweep.
#
# 8 dt values x 7 noise levels = 56 GPU tasks.
# Each task runs all samples in one process with batched Hessians on one A100 MIG slice.

set -euo pipefail

module purge
module load StdEnv/2023 python/3.11 cuda/12.6

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
N_STEPS="${N_STEPS:-2000}"
START_FROM="${START_FROM:-minimum_noised}"
OUT_ROOT="${OUT_ROOT:-$SCRATCH/gadplus/runs/lj_batched_gad_${SLURM_ARRAY_JOB_ID}}"
MAX_ATOM_DISP="${MAX_ATOM_DISP:-0.05}"
MAX_ATOM_DISP_START="${MAX_ATOM_DISP_START:-}"
MAX_ATOM_DISP_GROWTH="${MAX_ATOM_DISP_GROWTH:-1.0}"
MASS="${MASS:-1.008}"
DESCENT_UNTIL_NNEG="${DESCENT_UNTIL_NNEG:--1}"
BLEND_SHARPNESS="${BLEND_SHARPNESS:-0.0}"

read -r -a DTS <<< "${DTS:-0.0005 0.001 0.002 0.003 0.005 0.007 0.010 0.020}"
read -r -a NOISES <<< "${NOISES:-0.00 0.01 0.03 0.05 0.10 0.15 0.20}"

DT_IDX=$(( SLURM_ARRAY_TASK_ID / ${#NOISES[@]} ))
NOISE_IDX=$(( SLURM_ARRAY_TASK_ID % ${#NOISES[@]} ))
DT=${DTS[$DT_IDX]}
NOISE=${NOISES[$NOISE_IDX]}

OUT_DIR="$OUT_ROOT/gad_dt${DT}/noise_${NOISE}"
mkdir -p "$OUT_DIR"

echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID dt=$DT noise=$NOISE n=$N_SAMPLES steps=$N_STEPS max_atom_disp=$MAX_ATOM_DISP max_atom_disp_start=${MAX_ATOM_DISP_START:-none} max_atom_disp_growth=$MAX_ATOM_DISP_GROWTH mass=$MASS descent_until_nneg=$DESCENT_UNTIL_NNEG blend_sharpness=$BLEND_SHARPNESS out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

EXTRA_ARGS=()
if [[ -n "$MAX_ATOM_DISP_START" ]]; then
  EXTRA_ARGS+=(--max-atom-disp-start "$MAX_ATOM_DISP_START")
fi

python -u scripts/lj_batched_gad_runner.py \
  --device cuda \
  --n-samples "$N_SAMPLES" \
  --n-steps "$N_STEPS" \
  --start-from "$START_FROM" \
  --noise "$NOISE" \
  --dt "$DT" \
  --max-atom-disp "$MAX_ATOM_DISP" \
  "${EXTRA_ARGS[@]}" \
  --max-atom-disp-growth "$MAX_ATOM_DISP_GROWTH" \
  --mass "$MASS" \
  --descent-until-nneg "$DESCENT_UNTIL_NNEG" \
  --blend-sharpness "$BLEND_SHARPNESS" \
  --output-dir "$OUT_DIR"

echo "done"
