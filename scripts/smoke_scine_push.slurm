#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/smoke_scine_push_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/smoke_scine_push_%A_%a.err
#SBATCH --array=0-3
#
# Push SCINE/DFTB0 GAD further: can we close the rest of the gap to HIP?
# 30 samples at 10pm. Configs explore more steps + adaptive variants.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

declare -a CONFIGS=(
    "dt005_20k 0.005 20000"
    "dt007_15k 0.007 15000"
    "adapt_dt005_dtmax010_15k 0.005 15000 --use-adaptive-dt --dt-max 0.010"
    "adapt_dt005_dtmax015_15k 0.005 15000 --use-adaptive-dt --dt-max 0.015"
)

IDX=$SLURM_ARRAY_TASK_ID
read -r LABEL DT NSTEPS EXTRA <<< "${CONFIGS[$IDX]}"

OUT="/lustre07/scratch/memoozd/gadplus/runs/smoke_scine_push_${SLURM_ARRAY_JOB_ID}/${LABEL}"
mkdir -p "$OUT"
echo "Task $IDX: $LABEL  dt=$DT nsteps=$NSTEPS extra='$EXTRA'"

python -u scripts/gad_smoke.py \
    --backend scine --method DFTB0 \
    --noise 0.10 --n-samples 30 \
    --n-steps "$NSTEPS" --dt "$DT" \
    --force-threshold 0.01 --force-criterion fmax \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUT" \
    $EXTRA
