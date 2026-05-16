#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/main_scine_dftb23_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/main_scine_dftb23_%A_%a.err
#SBATCH --array=0-11
#
# Full SCINE/DFTB2 and DFTB3 grids at the headline recipe (dt=0.007, 15k).
# Array layout: tid = method_idx * 6 + noise_idx
#   method_idx 0 = DFTB2
#   method_idx 1 = DFTB3
# 287 T1x test samples × 6 noise levels each.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

NOISES_PM=(10 30 50 100 150 200)
NOISES_A=(0.10 0.30 0.50 1.00 1.50 2.00)
METHODS=(DFTB2 DFTB3)

METHOD_IDX=$((SLURM_ARRAY_TASK_ID / 6))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % 6))
METHOD=${METHODS[$METHOD_IDX]}
NOISE_PM=${NOISES_PM[$NOISE_IDX]}
NOISE_A=${NOISES_A[$NOISE_IDX]}

OUT="/lustre07/scratch/memoozd/gadplus/runs/main_scine_${METHOD,,}_15k_${SLURM_ARRAY_JOB_ID}/noise${NOISE_PM}pm"
mkdir -p "$OUT"
echo "Task $SLURM_ARRAY_TASK_ID | $METHOD | noise=${NOISE_PM}pm"

python -u scripts/gad_smoke.py \
    --backend scine --method "$METHOD" \
    --noise "$NOISE_A" --n-samples 287 \
    --n-steps 15000 --dt 0.007 \
    --force-threshold 0.01 --force-criterion fmax \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUT"
