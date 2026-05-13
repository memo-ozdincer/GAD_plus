#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/smoke_scine_neck15k_%A.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/smoke_scine_neck15k_%A.err
#
# no-Eckart ablation at 15k budget, single 10pm probe.
# 30 samples to see if the original "Eckart vs no-Eckart ±3pp" holds at
# the new budget.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

OUT="/lustre07/scratch/memoozd/gadplus/runs/smoke_scine_neck15k_${SLURM_JOB_ID}"
mkdir -p "$OUT"

python -u scripts/gad_smoke.py \
    --backend scine --method DFTB0 \
    --noise 0.10 --n-samples 30 \
    --n-steps 15000 --dt 0.007 \
    --no-projection \
    --force-threshold 0.01 --force-criterion fmax \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUT"
