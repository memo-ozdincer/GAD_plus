#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/irc_canon_test_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/irc_canon_test_%A_%a.err
#SBATCH --array=0-5
#
# Test-split sella_hip IRC TOPO sweep on the canonical noised-TS GAD runs.
# Closes the train-vs-test paper-hygiene gap (existing IRC validations are
# train n=300; this gives n=287 test-split numbers).
#
# Tasks (noise pm): 0=10  1=30  2=50  3=100  4=150  5=200

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export WANDB_DISABLED=true PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg

NOISES=(10 30 50 100 150 200)
NPM=${NOISES[$SLURM_ARRAY_TASK_ID]}

SURVEY=/lustre07/scratch/memoozd/gadplus/runs/test_set/gad_dt003_fmax
OUT=/lustre07/scratch/memoozd/gadplus/runs/irc_test_gad_eckart_fmax/noise${NPM}pm
mkdir -p "$OUT" /lustre07/scratch/memoozd/gadplus/logs

python -u scripts/irc_validate.py \
    --survey-dir "$SURVEY" \
    --output-dir "$OUT" \
    --source-method gad_dt003_fmax \
    --method sella_hip \
    --noise-pm "$NPM" \
    --max-validate 287 \
    --n-dataset-samples 287 \
    --split test \
    --irc-steps 500 \
    --rmsd-threshold 0.3
