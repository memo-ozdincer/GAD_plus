#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/ircsellatest_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/ircsellatest_%A_%a.err
#SBATCH --array=0-5

# STAGE 2 for the Sella 5-tier split: IRC re-validation of the test-287 Sella
# baseline TSs (stage 1), all endpoints. Submit with
#   --dependency=afterok:<stage1_jobid>
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

NOISES=(10 30 50 100 150 200)
NOISE=${NOISES[$SLURM_ARRAY_TASK_ID]}
mkdir -p /lustre07/scratch/memoozd/gadplus/runs/irc_sella_test287

python -u scripts/irc_validate.py \
    --method sella_hip --source-method sella_cartesian_eckart_fmax0p01 \
    --coords-source summary --all-endpoints \
    --noise-pm "$NOISE" --max-validate 300 --irc-steps 500 --rmsd-threshold 0.3 \
    --survey-dir /lustre07/scratch/memoozd/gadplus/runs/sella_libdef_test287 \
    --n-dataset-samples 287 --split test \
    --output-dir /lustre07/scratch/memoozd/gadplus/runs/irc_sella_test287 \
    --no-write-viewer-bundles
