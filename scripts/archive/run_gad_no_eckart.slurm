#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/gadnoec_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/gadnoec_%A_%a.err
#SBATCH --array=0-5

# GAD without Eckart projection, dt=0.003, 2000 steps.
# New fmax<0.01 convergence criterion (fair comparison with Sella).

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)
NOISE=${NOISES[$SLURM_ARRAY_TASK_ID]}

mkdir -p /lustre07/scratch/memoozd/gadplus/runs/gad_no_eckart

python -u scripts/method_single.py \
    --method gad_dt003_no_eckart \
    --noise "$NOISE" \
    --n-samples 300 \
    --n-steps 2000 \
    --output-dir /lustre07/scratch/memoozd/gadplus/runs/gad_no_eckart
