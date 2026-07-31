#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/mace_compat_%A.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/mace_compat_%A.err

set -euo pipefail

module purge
module load StdEnv/2023 python/3.11 cuda/12.6

PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

OUT="${OUT:-/lustre07/scratch/memoozd/gadplus/runs/mace_compatibility_smoke_${SLURM_JOB_ID}.parquet}"
MODEL="${MODEL:-small}"
SAMPLE_IDS="${SAMPLE_IDS:-0,1,2}"

python -u scripts/mace_compatibility_smoke.py \
  --output "$OUT" \
  --model "$MODEL" \
  --device cuda \
  --sample-ids "$SAMPLE_IDS"
