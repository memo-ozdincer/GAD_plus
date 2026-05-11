#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/irchybridretry_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/irchybridretry_%A_%a.err
#SBATCH --array=0-1

# Retry of IRC cells 8 and 9 from the original SLURM 60699659.
# Original failed because :g format truncates 0.10 -> 0.1 in the parquet
# filename, but the SLURM script passed "0.10" as the source-method tag.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

NOISE_PM_ARR=("10" "100")
NPM=${NOISE_PM_ARR[$SLURM_ARRAY_TASK_ID]}

SURVEY_DIR=/lustre07/scratch/memoozd/gadplus/runs/hybrid_for_irc/hybrid_swfalse_dt5e-3_tr0.10_${NPM}pm
OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/irc_hybrid/hybrid_swfalse_dt5e-3_tr0.10_${NPM}pm
mkdir -p "$OUTDIR"

echo "task=$SLURM_ARRAY_TASK_ID survey=$SURVEY_DIR noise=${NPM}pm"

python -u scripts/irc_validate.py \
    --method sella_hip --source-method "hybrid_dt0.005_tr0.1" \
    --coords-source summary --all-endpoints \
    --noise-pm "$NPM" --max-validate 300 --irc-steps 500 --rmsd-threshold 0.3 \
    --survey-dir "$SURVEY_DIR" \
    --n-dataset-samples 300 --split test \
    --output-dir "$OUTDIR" \
    --no-write-viewer-bundles
