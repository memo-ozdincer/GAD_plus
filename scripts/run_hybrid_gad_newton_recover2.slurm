#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/hgnrec2_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/hgnrec2_%A_%a.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# All cells we still need (excludes 20-24, 40-43 which already have summaries):
# 0-19, 25-29, 30-39, 44-49 (cells 0-6 already covered by 60403827; skip them here)
CELLS=(
  "hybrid:false"
  "hybrid_eckart:false"
  "hybrid_eckart:true"
  "hybrid_damped_eckart:false"
  "hybrid_damped_eckart:true"
)
NOISES=(0.01 0.10)
TRUST_RADII=(0.005 0.01 0.02 0.05 0.10)

T=$SLURM_ARRAY_TASK_ID
CELL_IDX=$((T / 10))
REM=$((T % 10))
NOISE_IDX=$((REM / 5))
TR_IDX=$((REM % 5))

CELL=${CELLS[$CELL_IDX]}
METHOD=$(echo "$CELL" | cut -d: -f1)
SWITCH=$(echo "$CELL" | cut -d: -f2)
NOISE=${NOISES[$NOISE_IDX]}
TR=${TRUST_RADII[$TR_IDX]}

NOISE_PM=$(python -c "print(int(round($NOISE*1000)))")
TAG="${METHOD}_sw${SWITCH}_dt5e-3_tr${TR}"
OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/hybrid_gad_newton/${TAG}_${NOISE_PM}pm
mkdir -p "$OUTDIR"

echo "task=$T method=$METHOD switch=$SWITCH noise=${NOISE_PM}pm trust=$TR"

python -u scripts/hybrid_gad_newton_runner.py \
    --method "$METHOD" \
    --switch-by-eig "$SWITCH" \
    --gad-dt 5e-3 \
    --trust-radius "$TR" \
    --noise "$NOISE" \
    --n-samples 287 \
    --n-steps 1000 \
    --split test \
    --output-dir "$OUTDIR"
