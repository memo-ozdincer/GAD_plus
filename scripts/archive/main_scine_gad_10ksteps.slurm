#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/main_scine_gad10k_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/main_scine_gad10k_%A_%a.err
#SBATCH --array=0-5
#
# SCINE/DFTB0 + plain GAD (Eckart, dt=0.005) with 10000-step budget.
# Smoke sweep showed 2000 steps caps SCINE GAD at strict-conv≈32% at 10pm
# due to slow geometric force-decay on the DFTB0 PES; 10000 steps recovers
# ~90% (matches HIP). Same Hessian, projection, mode-tracking as before.
#
# Array layout: noise level (0=10pm ... 5=200pm).
# 287 T1x test samples each, n_steps=10000, fmax<0.01 ∧ n_neg=1 criterion.

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

OUT="/lustre07/scratch/memoozd/gadplus/runs/main_scine_gad10k_${SLURM_ARRAY_JOB_ID}/noise${NOISE_PM}pm"
mkdir -p "$OUT"
echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname) | noise=${NOISE_PM}pm"
echo "Out: $OUT"

python -u scripts/gad_smoke.py \
    --backend scine --method DFTB0 \
    --noise "$NOISE_A" --n-samples 287 \
    --n-steps 10000 --dt 0.005 \
    --force-threshold 0.01 --force-criterion fmax \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUT"
