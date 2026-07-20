#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --job-name=painn-native-tight
#SBATCH --gpus=a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/painn_native_tight_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/painn_native_tight_%j.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6

PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
OUT="/lustre07/scratch/memoozd/gadplus/runs/painn-native-tight-${SLURM_JOB_ID}"
mkdir -p "$OUT" /lustre07/scratch/memoozd/gadplus/logs
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -u scripts/build_painn_native_saddles.py \
    --n-samples 3 \
    --endpoint-fmax 0.005 \
    --neb-fmax 0.01 \
    --endpoint-max-steps 500 \
    --neb-max-steps 500 \
    --output-dir "$OUT"
