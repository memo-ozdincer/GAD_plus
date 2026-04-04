#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/smoke_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/smoke_%j.err
#
# Smoke test on Trillium (H100)

set -euo pipefail

module purge
module load StdEnv/2023 python/3.11.5 cuda/12.6

PROJECT_DIR="/project/rrg-aspuru/memoozd/GAD_plus"
source /project/rrg-aspuru/memoozd/ts-tools/.venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export TMPDIR="/scratch/memoozd/gadplus/tmp"
mkdir -p "$TMPDIR" /scratch/memoozd/gadplus/logs

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

python "$PROJECT_DIR/scripts/smoke_test.py"
