#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/smoke_xtb_gfn1_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/smoke_xtb_gfn1_%j.err
#
# Smoke: xTB GFN1 with HIP-like dt=0.005 (since GFN1 Hessian magnitudes
# match DFTB0). GAD + Sella, 5 samples, 500 steps, 10pm noise.
# Goal: GFN1 should behave like DFTB0 — high "almost_converged" rate from GAD,
# fast convergence from Sella.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

BASE="/lustre07/scratch/memoozd/gadplus/runs/smoke_xtb_gfn1_${SLURM_JOB_ID}"

echo "=== xTB/GFN1 + plain GAD (10pm, 500 steps, dt=0.005) ==="
python -u scripts/gad_smoke.py \
    --backend xtb --method gfn1 \
    --noise 0.10 --n-samples 5 \
    --n-steps 500 --dt 0.005 \
    --force-threshold 0.01 --force-criterion fmax \
    --split test --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$BASE/gad"

echo; echo "=== xTB/GFN1 + Sella libdef (10pm, 500 steps) ==="
python -u scripts/sella_smoke.py \
    --backend xtb --method gfn1 \
    --noise 0.10 --n-samples 5 \
    --max-steps 500 --fmax 0.01 --apply-eckart \
    --delta0 0.1 --gamma 0.4 --diag-every 1 \
    --split test --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$BASE/sella"
