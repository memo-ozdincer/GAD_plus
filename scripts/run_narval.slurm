#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/%j.err
#
# GAD_plus on Narval — single MIG slice job
#
# Submit:   sbatch scripts/run_narval.slurm
# Monitor:  squeue -u $USER
# Results:  python scripts/analyze.py /lustre07/scratch/memoozd/gadplus/runs/

set -euo pipefail

# ---- Modules ----
module purge
module load StdEnv/2023 python/3.11 cuda/12.6

# ---- Env ----
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
SCRATCH="/lustre07/scratch/memoozd"
source "$PROJECT_DIR/.venv/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

# Threading: pin to avoid contention on shared nodes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg
export TMPDIR="$SCRATCH/gadplus/tmp"
export MLFLOW_TRACKING_URI="file://$SCRATCH/gadplus/mlruns"
mkdir -p "$TMPDIR"

# ---- Run ----
echo "Job $SLURM_JOB_ID on $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"

python -m gadplus.orchestration.run \
    search=gad_projected \
    starting=noised_ts \
    max_samples=50 \
    output_dir="$SCRATCH/gadplus/runs/${SLURM_JOB_ID}"
