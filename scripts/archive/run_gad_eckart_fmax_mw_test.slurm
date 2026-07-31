#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/gadfmaxmwtest_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/gadfmaxmwtest_%A_%a.err
#SBATCH --array=0-5

# Mass-weighted-step gad_dt003_fmax on the T1x TEST split (287 samples) — the
# MW baseline for the test-split A/B. Runs from the worktree pinned at the
# pre-Cartesian commit be59e16 (sqrt_m * dq), with PYTHONPATH shadowing the
# editable install so it imports the MW gadplus. Venv (deps) reused from main.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
WT=/lustre07/scratch/memoozd/gadplus/worktrees/mw_baseline
cd "$WT"
source /lustre06/project/6033559/memoozd/GAD_plus/.venv/bin/activate
export PYTHONPATH="$WT/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)
NOISE=${NOISES[$SLURM_ARRAY_TASK_ID]}

mkdir -p /lustre07/scratch/memoozd/gadplus/runs/gad_eckart_fmax_mw_test

python -u scripts/method_single.py \
    --method gad_dt003_fmax \
    --split test \
    --noise "$NOISE" \
    --n-samples 287 \
    --n-steps 2000 \
    --output-dir /lustre07/scratch/memoozd/gadplus/runs/gad_eckart_fmax_mw_test
