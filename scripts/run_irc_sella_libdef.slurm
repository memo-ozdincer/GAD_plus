#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/irclibdef_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/irclibdef_%A_%a.err
#SBATCH --array=0-5

# IRC validation on the tuned-Sella (libdef: delta0=0.1, gamma=0.4)
# summaries from sella_tune. n=100 per cell. Used to populate the IRC
# row of the new tex with the "best Sella" canonical numbers.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

NOISES=(10 30 50 100 150 200)
NOISE=${NOISES[$SLURM_ARRAY_TASK_ID]}

OUT=/lustre07/scratch/memoozd/gadplus/runs/irc_sella_libdef
mkdir -p "$OUT"

python -u scripts/irc_validate.py \
    --method sella_hip \
    --source-method sella_cartesian_eckart_fmax0p01_libdef \
    --coords-source summary --all-endpoints \
    --noise-pm "$NOISE" --max-validate 100 --irc-steps 500 --rmsd-threshold 0.3 \
    --survey-dir /lustre07/scratch/memoozd/gadplus/runs/sella_tune/libdef \
    --n-dataset-samples 100 \
    --output-dir "$OUT" --no-write-viewer-bundles
