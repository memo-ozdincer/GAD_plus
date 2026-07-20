#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/labelled_ts_0pm_%A.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/labelled_ts_0pm_%A.err

# Two metrics on the T1x-labelled TS, test split n=287:
#   (1) True-overlap rate (no steps) — eval HIP at pos_transition once.
#   (2) True-convergence rate (full GAD) — start GAD from the labelled TS,
#       canonical gad_dt003_fmax recipe, 2000-step budget, fmax<0.01.
#
# Both written under runs/labelled_ts_0pm/.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

OUT=/lustre07/scratch/memoozd/gadplus/runs/labelled_ts_0pm
mkdir -p "$OUT" /lustre07/scratch/memoozd/gadplus/logs

echo "============================================================"
echo "[1/2] LABELLED-TS TRUE-OVERLAP RATE (no optimisation steps)"
echo "============================================================"
python -u scripts/eval_labelled_ts.py \
    --split test \
    --n-samples 287 \
    --force-threshold 0.01 \
    --neg-eig-threshold 1e-4 \
    --output-dir "$OUT"

echo
echo "============================================================"
echo "[2/2] LABELLED-TS TRUE-CONVERGENCE RATE (full GAD, 2000 steps)"
echo "============================================================"
python -u scripts/method_single.py \
    --method gad_dt003_fmax \
    --start-from ts_noised \
    --noise 0.0 \
    --n-samples 287 \
    --n-steps 2000 \
    --force-criterion fmax \
    --force-threshold 0.01 \
    --split test \
    --output-dir "$OUT"
