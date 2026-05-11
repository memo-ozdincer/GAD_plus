#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/smoke_scine_gad_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/smoke_scine_gad_%j.err
#
# Tier-1 smoke: gad_projected on SCINE/DFTB0.
# 5 test-split samples, noise=1.0 A (100 pm), 500 steps, 16 parallel workers.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
# Per-worker BLAS threads; pool size is set via --n-workers below.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

OUT="/lustre07/scratch/memoozd/gadplus/runs/smoke_tier1_scine_gad_${SLURM_JOB_ID}"
mkdir -p "$OUT"
echo "Job $SLURM_JOB_ID on $(hostname) | CPUs: ${SLURM_CPUS_PER_TASK}"

python -u scripts/gad_smoke.py \
    --backend scine --method DFTB0 \
    --noise 1.0 --n-samples 5 \
    --n-steps 500 --dt 0.003 \
    --force-threshold 0.01 --force-criterion fmax \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUT"
