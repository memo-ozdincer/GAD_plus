#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/testsellarec_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/testsellarec_%A_%a.err
#SBATCH --array=0-5

# Recovery for the 6 high-noise Sella cells that timed out / OOMed in 60007652.
# Increased to 96G mem, 12h time.
#   0,1: carteck_default 150,200pm
#   2:   carteck_libdef 200pm
#   3,4,5: sella_internal_default 100,150,200pm

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# (tag, delta0, gamma, coords_flag, noise)
JOBS=(
    "carteck_default:0.048:0.0:--cartesian --apply-eckart:0.15"
    "carteck_default:0.048:0.0:--cartesian --apply-eckart:0.20"
    "carteck_libdef:0.1:0.4:--cartesian --apply-eckart:0.20"
    "internal_default:0.048:0.0::0.10"
    "internal_default:0.048:0.0::0.15"
    "internal_default:0.048:0.0::0.20"
)

J=${JOBS[$SLURM_ARRAY_TASK_ID]}
TAG=$(echo "$J" | cut -d: -f1)
DELTA0=$(echo "$J" | cut -d: -f2)
GAMMA=$(echo "$J" | cut -d: -f3)
COORDS=$(echo "$J" | cut -d: -f4)
NOISE=$(echo "$J" | cut -d: -f5)

OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_set/sella_${TAG}
mkdir -p "$OUTDIR"

echo "task=$SLURM_ARRAY_TASK_ID tag=$TAG noise=$NOISE coords=[$COORDS]"

python -u scripts/sella_baseline.py \
    --fmax 0.01 \
    --noise "$NOISE" \
    --n-samples 287 \
    --max-steps 2000 \
    --split test \
    $COORDS \
    --delta0 "$DELTA0" \
    --gamma "$GAMMA" \
    --config-tag "$TAG" \
    --output-dir "$OUTDIR"
