#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --job-name=hip-hessian-sub
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/hip_hessian_sub_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/hip_hessian_sub_%j.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
OUT="/lustre07/scratch/memoozd/gadplus/runs/hip-hessian-sub-${SLURM_JOB_ID}"
mkdir -p "$OUT" /lustre07/scratch/memoozd/gadplus/logs
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="/lustre07/scratch/memoozd/gadplus/python-overrides:$PROJECT_DIR/src:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
python -u scripts/hip_hessian_source_substitution_smoke.py --output-dir "$OUT" "$@"
