#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/sweep_dt_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/sweep_dt_%j.err
#
# Phase 1: Parameter sweep for pure GAD.
# Grid: dt=[0.001,0.003,0.005,0.01,0.02] x k_track=[0,4,8]
# 10 test samples, 100 steps, noise=0.05A
#
# Submit:  sbatch scripts/run_sweep_dt.slurm

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
mkdir -p "$TMPDIR" "$SCRATCH/gadplus/logs" "$SCRATCH/gadplus/runs/sweep_dt"

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

python "$PROJECT_DIR/scripts/sweep_dt.py" \
    --n-samples 10 \
    --n-steps 100 \
    --noise 0.05 \
    --split test \
    --output-dir "$SCRATCH/gadplus/runs/sweep_dt"

echo "Done."
