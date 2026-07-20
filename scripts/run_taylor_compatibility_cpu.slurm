#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --job-name=cpu-taylor
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/cpu_taylor_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/cpu_taylor_%j.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11

PROJECT_DIR=/lustre06/project/6033559/memoozd/GAD_plus
BACKEND="${BACKEND:-scine}"
OUT=/lustre07/scratch/memoozd/gadplus/runs/${BACKEND}-taylor-compatibility-${SLURM_JOB_ID}
mkdir -p "$OUT" /lustre07/scratch/memoozd/gadplus/logs
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -u scripts/taylor_compatibility.py \
  --backend "$BACKEND" \
  --output-dir "$OUT" \
  ${TAYLOR_ARGS:-}
