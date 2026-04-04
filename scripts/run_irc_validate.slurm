#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/irc_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/irc_%j.err
#
# Phase 5: IRC validation on converged TS from Phase 2.
# Validates first 10 converged TS at noise=10pm.
#
# Submit:  sbatch scripts/run_irc_validate.slurm

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
mkdir -p "$TMPDIR" "$SCRATCH/gadplus/logs" "$SCRATCH/gadplus/runs/irc_validation"

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u "$PROJECT_DIR/scripts/irc_validate.py" \
    --noise-pm 10 \
    --max-validate 10 \
    --irc-steps 100 \
    --rmsd-threshold 0.3

echo "Done."
