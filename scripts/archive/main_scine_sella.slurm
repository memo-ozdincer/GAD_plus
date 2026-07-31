#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/main_scine_sella_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/main_scine_sella_%A_%a.err
#SBATCH --array=0-5
#
# SCINE/DFTB0 + Sella libdef (Cartesian + Eckart, delta0=0.1, gamma=0.4,
# Hessian every step). Recreates the "Sella libdef" row of
# analysis_2026_04_29/noise_sweep_with_irc.csv with a non-ML Hessian.
#
# Array layout: noise level (0=10pm, 1=30pm, 2=50pm, 3=100pm, 4=150pm, 5=200pm).
# 287 T1x test samples each, 2000 steps, fmax<0.01 ∧ n_neg=1 criterion.

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
NOISE_PM=${NOISES_PM[$SLURM_ARRAY_TASK_ID]}
NOISE_A=${NOISES_A[$SLURM_ARRAY_TASK_ID]}

OUT="/lustre07/scratch/memoozd/gadplus/runs/main_scine_sella_${SLURM_ARRAY_JOB_ID}/noise${NOISE_PM}pm"
mkdir -p "$OUT"
echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname) | noise=${NOISE_PM}pm"
echo "Out: $OUT"

python -u scripts/sella_smoke.py \
    --backend scine --method DFTB0 \
    --noise "$NOISE_A" --n-samples 287 \
    --max-steps 2000 --fmax 0.01 \
    --apply-eckart \
    --delta0 0.1 --gamma 0.4 --diag-every 1 \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUT"
