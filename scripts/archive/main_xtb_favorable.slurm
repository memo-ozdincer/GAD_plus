#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/main_xtb_fav_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/main_xtb_fav_%A_%a.err
#SBATCH --array=0-1
#
# xTB/GFN1 panel on the top-30 xTB-favorable T1x test samples (those where
# GFN1 fmax at HIP-TS is lowest, i.e., xTB's saddle is nearest HIP's).
# Array layout: 0 = GAD, 1 = Sella libdef (cart + Eckart).
# Single noise level (10 pm) — pure feasibility check.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

INDICES=$(cat analysis_2026_04_29/xtb_favorable_top30.txt)
echo "xTB-favorable indices: $INDICES"

OUT="/lustre07/scratch/memoozd/gadplus/runs/main_xtb_favorable_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$OUT"

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    echo "=== xTB/GFN1 + plain GAD, top-30 favorable, 10pm, 500 steps, dt=0.005 ==="
    python -u scripts/gad_smoke.py \
        --backend xtb --method gfn1 \
        --noise 0.10 --sample-indices "$INDICES" \
        --n-steps 500 --dt 0.005 \
        --force-threshold 0.01 --force-criterion fmax \
        --split test --n-workers "$SLURM_CPUS_PER_TASK" \
        --output-dir "$OUT/gad"
else
    echo "=== xTB/GFN1 + Sella libdef, top-30 favorable, 10pm, 500 steps ==="
    python -u scripts/sella_smoke.py \
        --backend xtb --method gfn1 \
        --noise 0.10 --sample-indices "$INDICES" \
        --max-steps 500 --fmax 0.01 --apply-eckart \
        --delta0 0.1 --gamma 0.4 --diag-every 1 \
        --split test --n-workers "$SLURM_CPUS_PER_TASK" \
        --output-dir "$OUT/sella"
fi
