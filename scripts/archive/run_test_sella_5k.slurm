#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/testsella5k_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/testsella5k_%A_%a.err
#SBATCH --array=0-11

# Sella with matched 5000-step budget (vs GAD's 5000) — for "fair comparison"
# headline. Also one 10000-step row at 200pm for extra-stretched case.
#
# 6 noise × 2 budgets (5000, 10000) = 12 cells × n=287.
# Uses Sella cart+Eckart with delta0=0.10, gamma=0.40.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)
N_IDX=$((SLURM_ARRAY_TASK_ID % 6))
B_IDX=$((SLURM_ARRAY_TASK_ID / 6))
NOISE=${NOISES[$N_IDX]}

if [ $B_IDX -eq 0 ]; then
    BUDGET=5000
    TAG=carteck_libdef_5k
else
    BUDGET=10000
    TAG=carteck_libdef_10k
fi

OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_sella_extended/${TAG}
mkdir -p "$OUTDIR"

echo "task=$SLURM_ARRAY_TASK_ID noise=$NOISE budget=$BUDGET tag=$TAG"

python -u scripts/sella_baseline.py --fmax 0.01 --noise "$NOISE" \
    --n-samples 287 --max-steps "$BUDGET" --split test \
    --cartesian --apply-eckart --delta0 0.1 --gamma 0.4 \
    --config-tag "$TAG" \
    --output-dir "$OUTDIR"
