#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/irchybtest_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/irchybtest_%A_%a.err
#SBATCH --array=0-5

# IRC re-validation of the headline hybrid (damped Eckart eig-switch tr=0.05) on
# the TEST split (n=287), all endpoints -> full 5-tier split per sample.
# Source TSs: hybrid_for_irc/ (already test-287, coords in summary).
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

NOISES=(10 30 50 100 150 200)
NPM=${NOISES[$SLURM_ARRAY_TASK_ID]}
TAG=hybrid_damped_eckart_swtrue_dt5e-3_tr0.05
SOURCE_METHOD=hybrid_damped_eckart_swEIG_dt0.005_tr0.05
SURVEY_DIR=/lustre07/scratch/memoozd/gadplus/runs/hybrid_for_irc/${TAG}_${NPM}pm
OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/irc_hybrid_test287/${NPM}pm
mkdir -p "$OUTDIR"

python -u scripts/irc_validate.py \
    --method sella_hip --source-method "$SOURCE_METHOD" \
    --coords-source summary --all-endpoints \
    --noise-pm "$NPM" --max-validate 300 --irc-steps 500 --rmsd-threshold 0.3 \
    --survey-dir "$SURVEY_DIR" \
    --n-dataset-samples 287 --split test \
    --output-dir "$OUTDIR" \
    --no-write-viewer-bundles
