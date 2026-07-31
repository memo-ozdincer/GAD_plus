#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/testsellanohess_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/testsellanohess_%A_%a.err
#SBATCH --array=0-11

# Sella WITHOUT Hessian injection on T1x test split (n=287). Two configs
# (cart+Eckart fallback to BFGS, and internal-coords fallback to BFGS),
# 6 noise levels = 12 array tasks. Tests whether Sella's Hessian-injection
# was carrying the comparison — paper argument needs Sella w/o Hessians
# to lose to GAD by a wider margin.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# (tag, coords_flag)
CONFIGS=(
    "carteck_nohess:--cartesian --apply-eckart"
    "internal_nohess:"
)
NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)

CFG_IDX=$((SLURM_ARRAY_TASK_ID / 6))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % 6))

CFG=${CONFIGS[$CFG_IDX]}
TAG=$(echo "$CFG" | cut -d: -f1)
COORDS=$(echo "$CFG" | cut -d: -f2)
NOISE=${NOISES[$NOISE_IDX]}

OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_set/sella_${TAG}
mkdir -p "$OUTDIR"

echo "task=$SLURM_ARRAY_TASK_ID tag=$TAG coords=[$COORDS] noise=$NOISE no-hessian"

python -u scripts/sella_baseline.py \
    --fmax 0.01 \
    --noise "$NOISE" \
    --n-samples 287 \
    --max-steps 2000 \
    --split test \
    $COORDS \
    --no-hessian \
    --delta0 0.1 --gamma 0.4 \
    --config-tag "$TAG" \
    --output-dir "$OUTDIR"
