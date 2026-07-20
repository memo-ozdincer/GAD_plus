#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --job-name=verify-gad-step
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/verify_gad_step_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/verify_gad_step_%j.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6

PROJECT_DIR=/lustre06/project/6033559/memoozd/GAD_plus
cd "$PROJECT_DIR"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/lustre07/scratch/memoozd/gadplus/deps/wandb:${PYTHONPATH:-}

python -u scripts/verify_historical_gad_step.py \
  --trajectory-dir /lustre07/scratch/memoozd/gadplus/runs/test_dtgrid/gad_dt007_fmax \
  --checkpoint /lustre06/project/6033559/memoozd/models/hip_v2.ckpt \
  --dataset /lustre06/project/6033559/memoozd/data/transition1x.h5 \
  --sample-ids 0 2 3 5 \
  --noise-label 150 \
  --dt 0.007
