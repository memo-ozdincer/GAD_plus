#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/sint150_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/sint150_%A_%a.err
#SBATCH --array=0-3
#
# Sella internal d=1 @ 150pm refill, 4-way partitioned, 8h each.
# Original cell timed out at n=172 with no summary parquet.
# When all 4 partitions land, pool them via build_pooled_summaries_2026_05_16.py
# (or a one-liner) and queue IRC validation.
#
# Partitions: [0,72), [72,144), [144,216), [216,287)

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

T=$SLURM_ARRAY_TASK_ID
RANGES=(0 72 144 216 287)
S=${RANGES[$T]}
E=${RANGES[$((T + 1))]}

OUT=/lustre07/scratch/memoozd/gadplus/runs/safetynet/sella_internal_default
mkdir -p "$OUT"

echo "T=$T samples [$S,$E) noise=150pm"

python -u scripts/sella_baseline.py --fmax 0.01 --noise 0.15 \
    --n-samples 287 --max-steps 2000 --split test \
    --sample-start "$S" --sample-end "$E" \
    --config-tag "internal_default" --output-dir "$OUT"

echo "DONE"
