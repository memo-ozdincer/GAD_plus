#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/testlowdt_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/testlowdt_%A_%a.err
#SBATCH --array=0-17

# Low-dt GAD on T1x test (n=287). Three small dts × 6 noise = 18 tasks.
# Step budget scaled inversely with dt to give roughly equal "physical
# time" coverage:
#   dt=0.001  -> 20000 steps
#   dt=0.0005 -> 40000 steps
#   dt=0.0001 -> 100000 steps  (may not converge — diagnostic)
# Hypothesis: smaller dt approaches true GAD ODE; if convergence fails
# even at very small dt, the sample is fundamentally GAD-unfriendly.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

CFGS=(
    "gad_dt001_fmax:20000"
    "gad_dt0005_fmax:40000"
    "gad_dt0001_fmax:100000"
)
NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)

CFG_IDX=$((SLURM_ARRAY_TASK_ID / 6))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % 6))
CFG=${CFGS[$CFG_IDX]}
METHOD=$(echo "$CFG" | cut -d: -f1)
STEPS=$(echo "$CFG" | cut -d: -f2)
NOISE=${NOISES[$NOISE_IDX]}

OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_lowdt/${METHOD}
mkdir -p "$OUTDIR"

echo "task=$SLURM_ARRAY_TASK_ID method=$METHOD noise=$NOISE steps=$STEPS"

python -u scripts/method_single.py \
    --method "$METHOD" \
    --noise "$NOISE" \
    --n-samples 287 \
    --n-steps "$STEPS" \
    --split test \
    --output-dir "$OUTDIR"
