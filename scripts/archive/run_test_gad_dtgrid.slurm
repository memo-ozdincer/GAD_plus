#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/testgaddt_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/testgaddt_%A_%a.err
#SBATCH --array=0-35

# GAD dt grid on T1x TEST split (n=287), 5000-step budget (vs 2000 in
# initial test sweep) — addresses step-cap bottleneck observed at high
# noise where 65% of dt=0.003 runs hit the 2000-step cap.
#
# 6 dt values × 6 noise levels = 36 tasks
#   dt: 0.003, 0.004, 0.005, 0.006, 0.007, 0.008
# Probes the stable–unstable cliff (dt=0.010 already known to collapse).

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

METHODS=(gad_dt003_fmax gad_dt004_fmax gad_dt005_fmax gad_dt006_fmax gad_dt007_fmax gad_dt008_fmax)
NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)

METHOD_IDX=$((SLURM_ARRAY_TASK_ID / 6))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % 6))
METHOD=${METHODS[$METHOD_IDX]}
NOISE=${NOISES[$NOISE_IDX]}

OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_dtgrid/${METHOD}
mkdir -p "$OUTDIR"

echo "task=$SLURM_ARRAY_TASK_ID method=$METHOD noise=$NOISE budget=5000 split=test"

python -u scripts/method_single.py \
    --method "$METHOD" \
    --noise "$NOISE" \
    --n-samples 287 \
    --n-steps 5000 \
    --split test \
    --output-dir "$OUTDIR"
