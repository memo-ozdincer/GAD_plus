#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --job-name=hip-field-check
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/hip_field_check_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/hip_field_check_%j.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6

PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
OUT="/lustre07/scratch/memoozd/gadplus/runs/hip-field-check-${SLURM_JOB_ID}"
mkdir -p "$OUT" /lustre07/scratch/memoozd/gadplus/logs
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="/lustre07/scratch/memoozd/gadplus/python-overrides:$PROJECT_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# HIP_FIELD_ARGS is used only for small dependency smokes; the default is the
# fixed 5x3 diagnostic recorded in the experiment manifest.
python -u scripts/hip_field_consistency.py --output-dir "$OUT" ${HIP_FIELD_ARGS:-}
