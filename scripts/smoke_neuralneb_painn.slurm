#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --job-name=painn-efh-smoke
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/painn_efh_smoke_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/painn_efh_smoke_%j.err

# One short backend-validation task. Do not use this allocation pattern for
# production molecule-level work; production jobs will pack many trajectories.
set -euo pipefail

module purge
module load StdEnv/2023 python/3.11

PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
RUN_DIR="/lustre07/scratch/memoozd/gadplus/runs/painn-efh-smoke-${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR" /lustre07/scratch/memoozd/gadplus/logs
cd "$PROJECT_DIR"

source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "job=$SLURM_JOB_ID host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u scripts/smoke_neuralneb_painn.py --device cuda | tee "$RUN_DIR/smoke.txt"
