#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/sellatune_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/sellatune_%A_%a.err
#SBATCH --array=0-17

# Sella parameter-tuning grid: 3 configs x 6 noise levels = 18 tasks.
# n=100 samples each (subset, fast). Output: separate parquets per config
# under sella_tune/<config_name>/. Compare against the canonical
# sella_2000 results (which use carteck_default).
#
# Configs:
#   carteck_default : delta0=0.048, gamma=0.0  (current "tuned" setting)
#   carteck_libdef  : delta0=0.1,   gamma=0.4
#   carteck_lson    : delta0=0.048, gamma=0.4  (line search ON, current trust)

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# (config_tag, delta0, gamma)
CONFIGS=(
    "default:0.048:0.0"
    "libdef:0.1:0.4"
    "lson:0.048:0.4"
)
NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)

CFG_IDX=$((SLURM_ARRAY_TASK_ID / 6))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % 6))

CFG=${CONFIGS[$CFG_IDX]}
TAG=$(echo "$CFG" | cut -d: -f1)
DELTA0=$(echo "$CFG" | cut -d: -f2)
GAMMA=$(echo "$CFG" | cut -d: -f3)
NOISE=${NOISES[$NOISE_IDX]}

OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/sella_tune/${TAG}
mkdir -p "$OUTDIR"

echo "task=$SLURM_ARRAY_TASK_ID tag=$TAG delta0=$DELTA0 gamma=$GAMMA noise=$NOISE"

python -u scripts/sella_baseline.py \
    --fmax 0.01 \
    --noise "$NOISE" \
    --n-samples 100 \
    --max-steps 2000 \
    --cartesian --apply-eckart \
    --delta0 "$DELTA0" \
    --gamma "$GAMMA" \
    --config-tag "$TAG" \
    --output-dir "$OUTDIR"
