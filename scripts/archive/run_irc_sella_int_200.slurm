#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/ircsi200_%A.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/ircsi200_%A.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

mkdir -p /lustre07/scratch/memoozd/gadplus/runs/irc_sella_int_2000

python -u scripts/irc_validate.py \
    --method sella_hip --source-method sella_internal_fmax0p01 \
    --coords-source summary --all-endpoints \
    --noise-pm 200 --max-validate 300 --irc-steps 500 --rmsd-threshold 0.3 \
    --survey-dir /lustre07/scratch/memoozd/gadplus/runs/sella_2000 \
    --n-dataset-samples 300 \
    --output-dir /lustre07/scratch/memoozd/gadplus/runs/irc_sella_int_2000 \
    --no-write-viewer-bundles
