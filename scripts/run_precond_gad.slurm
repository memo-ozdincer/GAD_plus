#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/precond_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/precond_%A_%a.err
#SBATCH --array=0-29
#
# Preconditioned GAD: 5 methods x 6 noise levels = 30 parallel jobs.
# 300 samples, 1000 steps each. One MIG slice per (method, noise) pair.
#
# Methods:
#   0 = gad_small_dt      (baseline control, dt=0.005)
#   1 = precond_gad_001   (eig_floor=0.01, dt=0.005)
#   2 = precond_gad_005   (eig_floor=0.05, dt=0.005)
#   3 = precond_gad_01    (eig_floor=0.1,  dt=0.005)
#   4 = precond_gad_dt01  (eig_floor=0.01, dt=0.01)
#
# Noise: 0=10pm 1=30pm 2=50pm 3=100pm 4=150pm 5=200pm
#
# Array mapping: task_id = method_idx * 6 + noise_idx
#
# Submit:  sbatch scripts/run_precond_gad.slurm

set -euo pipefail

module purge
module load StdEnv/2023 python/3.11 cuda/12.6

PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
SCRATCH="/lustre07/scratch/memoozd"
source "$PROJECT_DIR/.venv/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export TMPDIR="$SCRATCH/gadplus/tmp"
mkdir -p "$TMPDIR" "$SCRATCH/gadplus/logs" "$SCRATCH/gadplus/runs/precond_gad"

METHODS=(gad_small_dt precond_gad_001 precond_gad_005 precond_gad_01 precond_gad_dt01)
NOISE_LEVELS=(0.01 0.03 0.05 0.1 0.15 0.2)

METHOD_IDX=$((SLURM_ARRAY_TASK_ID / 6))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % 6))
METHOD=${METHODS[$METHOD_IDX]}
NOISE=${NOISE_LEVELS[$NOISE_IDX]}

echo "Job $SLURM_JOB_ID, task $SLURM_ARRAY_TASK_ID: method=$METHOD noise=$NOISE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u "$PROJECT_DIR/scripts/method_single.py" \
    --method "$METHOD" \
    --noise "$NOISE" \
    --n-samples 300 \
    --n-steps 1000 \
    --split train \
    --output-dir "$SCRATCH/gadplus/runs/precond_gad"

echo "Done."
