#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/testirc_smoke_%A.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/testirc_smoke_%A.err

# Smoke test: 5 samples to verify IRC is no longer silently failing
# after the split=test fix and exception traceback addition.
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

python -u scripts/irc_validate.py \
    --method sella_hip --source-method gad_dt007_fmax \
    --coords-source traj --all-endpoints --split test \
    --noise-pm 10 --max-validate 5 --irc-steps 200 --rmsd-threshold 0.3 \
    --survey-dir /lustre07/scratch/memoozd/gadplus/runs/test_dtgrid/gad_dt007_fmax \
    --n-dataset-samples 287 \
    --output-dir /tmp/irc_smoketest --no-write-viewer-bundles
