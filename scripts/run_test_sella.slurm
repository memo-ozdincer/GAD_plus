#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/testsella_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/testsella_%A_%a.err
#SBATCH --array=0-17

# Sella on T1x TEST split (287 samples). Three configs x 6 noise = 18 tasks.
#   sella_carteck_default : current (delta0=0.048, gamma=0)
#   sella_carteck_libdef  : tuning winner (delta0=0.1, gamma=0.4)
#   sella_internal_default: Sella default internal coords (no Eckart)

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# (config_tag, delta0, gamma, coord_flag)
# coord_flag is added to argv literally
CONFIGS=(
    "carteck_default:0.048:0.0:--cartesian --apply-eckart"
    "carteck_libdef:0.1:0.4:--cartesian --apply-eckart"
    "internal_default:0.048:0.0:"
)
NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)

CFG_IDX=$((SLURM_ARRAY_TASK_ID / 6))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % 6))

CFG=${CONFIGS[$CFG_IDX]}
TAG=$(echo "$CFG" | cut -d: -f1)
DELTA0=$(echo "$CFG" | cut -d: -f2)
GAMMA=$(echo "$CFG" | cut -d: -f3)
COORDS=$(echo "$CFG" | cut -d: -f4)
NOISE=${NOISES[$NOISE_IDX]}

OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_set/sella_${TAG}
mkdir -p "$OUTDIR"

echo "task=$SLURM_ARRAY_TASK_ID tag=$TAG delta0=$DELTA0 gamma=$GAMMA coords=[$COORDS] noise=$NOISE split=test"

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
