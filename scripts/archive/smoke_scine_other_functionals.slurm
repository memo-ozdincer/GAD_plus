#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/smoke_scine_alt_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/smoke_scine_alt_%A_%a.err
#SBATCH --array=0-3
#
# Validate the 10k-step recipe on other SCINE semiempirical methods.
# 20 T1x test samples each at 10pm noise. Same dt=0.005, n_steps=10000
# settings that gave 90% on DFTB0.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

METHODS=(DFTB2 DFTB3 PM6 AM1)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

OUT="/lustre07/scratch/memoozd/gadplus/runs/smoke_scine_alt_${SLURM_ARRAY_JOB_ID}/${METHOD}"
mkdir -p "$OUT"
echo "Task $SLURM_ARRAY_TASK_ID: SCINE/$METHOD"
echo "  Out: $OUT"

python -u scripts/gad_smoke.py \
    --backend scine --method "$METHOD" \
    --noise 0.10 --n-samples 20 \
    --n-steps 10000 --dt 0.005 \
    --force-threshold 0.01 --force-criterion fmax \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUT"
