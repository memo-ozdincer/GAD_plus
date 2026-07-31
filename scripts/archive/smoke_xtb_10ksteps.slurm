#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/smoke_xtb10k_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/smoke_xtb10k_%A_%a.err
#SBATCH --array=0-1
#
# Retry xTB-GFN1/GFN2 with the 10k-step recipe that fixed SCINE.
# Top-30 favorable samples at 10pm noise. Previous 500-step probe gave
# 0/30; if the failure is step-budget rather than PES-disagreement,
# this should show improvement.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

METHODS=(gfn1 gfn2)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

# Top-30 xTB-favorable samples (lowest fmax at HIP-TS+0 noise)
INDICES=$(tr '\n' ',' < /lustre06/project/6033559/memoozd/GAD_plus/analysis_2026_04_29/xtb_favorable_top30.txt | sed 's/,$//')

OUT="/lustre07/scratch/memoozd/gadplus/runs/smoke_xtb10k_${SLURM_ARRAY_JOB_ID}/${METHOD}"
mkdir -p "$OUT"
echo "Method: $METHOD"
echo "Indices: $INDICES"

python -u scripts/gad_smoke.py \
    --backend xtb --method "$METHOD" \
    --noise 0.10 \
    --sample-indices "$INDICES" \
    --n-steps 10000 --dt 0.005 \
    --force-threshold 0.01 --force-criterion fmax \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUT"
