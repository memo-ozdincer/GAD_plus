#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/gadbigdt_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/gadbigdt_%A_%a.err
#SBATCH --array=0-17

# Bigger GAD dt sweep (2026-04-28). Three dt values x 6 noise levels = 18 tasks.
# Same Eckart-projected GAD dynamics as the canonical gad_dt003_fmax, just
# larger step. Tests whether the GAD-vs-Sella comparison was unfair because
# GAD's per-step displacement (~1e-3 A at dt=0.003) is order-of-magnitude
# smaller than Sella's QN trust-region step (~1e-2 A).
#
# Array index decoding: idx = method_idx * 6 + noise_idx
#   methods: gad_dt005_fmax, gad_dt010_fmax, gad_dt020_fmax
#   noises:  0.01 0.03 0.05 0.10 0.15 0.20

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

METHODS=(gad_dt005_fmax gad_dt010_fmax gad_dt020_fmax)
NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)

METHOD_IDX=$((SLURM_ARRAY_TASK_ID / 6))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % 6))

METHOD=${METHODS[$METHOD_IDX]}
NOISE=${NOISES[$NOISE_IDX]}

OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/gad_bigger_dt/${METHOD}
mkdir -p "$OUTDIR"

echo "task=$SLURM_ARRAY_TASK_ID method=$METHOD noise=$NOISE outdir=$OUTDIR"

python -u scripts/method_single.py \
    --method "$METHOD" \
    --noise "$NOISE" \
    --n-samples 300 \
    --n-steps 2000 \
    --output-dir "$OUTDIR"
