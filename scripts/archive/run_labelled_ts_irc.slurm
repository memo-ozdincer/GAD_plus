#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/labelled_ts_irc_%A.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/labelled_ts_irc_%A.err
#
# IRC TOPO validation on the 253 converged TSs from
# runs/labelled_ts_0pm/summary_gad_dt003_fmax_0pm.parquet.
# Canonical rigorous (Hratchian-Schlegel) IRC, test split.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export WANDB_DISABLED=true PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg

SURVEY=/lustre07/scratch/memoozd/gadplus/runs/labelled_ts_0pm
OUT=$SURVEY/irc_rigorous
mkdir -p "$OUT" /lustre07/scratch/memoozd/gadplus/logs

python -u scripts/irc_validate.py \
    --survey-dir "$SURVEY" \
    --output-dir "$OUT" \
    --source-method gad_dt003_fmax \
    --method rigorous \
    --noise-pm 0 \
    --max-validate 287 \
    --n-dataset-samples 287 \
    --split test \
    --irc-steps 500 \
    --rmsd-threshold 0.3
