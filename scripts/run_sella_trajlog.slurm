#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/sellatraj_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/sellatraj_%A_%a.err
#SBATCH --array=0-2

# Sella with per-step trajectory logging at one noise level (100pm),
# n=50 samples. Three configs in array: (1) cart+Eckart current,
# (2) cart+Eckart, delta0=0.10, gamma=0.40, (3) cart+Eckart
# with line search ON (gamma=0.4). Used for step-size-vs-step figure
# and for the tuning comparison.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# (config_name, delta0, gamma)
CONFIGS=(
    "carteck_default:0.048:0.0"
    "carteck_libdef:0.1:0.4"
    "carteck_lson:0.048:0.4"
)
CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
NAME=$(echo "$CFG" | cut -d: -f1)
DELTA0=$(echo "$CFG" | cut -d: -f2)
GAMMA=$(echo "$CFG" | cut -d: -f3)

OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/sella_trajlog/${NAME}
mkdir -p "$OUTDIR"

echo "task=$SLURM_ARRAY_TASK_ID config=$NAME delta0=$DELTA0 gamma=$GAMMA"

python -u scripts/sella_trajlog.py \
    --noise 0.10 \
    --n-samples 50 \
    --max-steps 2000 \
    --fmax 0.01 \
    --cartesian --apply-eckart \
    --delta0 "$DELTA0" \
    --gamma "$GAMMA" \
    --config-name "$NAME" \
    --output-dir "$OUTDIR"
