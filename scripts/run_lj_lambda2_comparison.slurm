#!/bin/bash
#SBATCH --job-name=lj_lambda2_compare
#SBATCH --account=def-aspuru
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/lj_lambda2_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/lj_lambda2_%j.err

set -euo pipefail

module purge
module load StdEnv/2023 python/3.11.5

WORK_ROOT=/scratch/memoozd/GAD/GAD_plus
RUN_ROOT=${RUN_ROOT:-/scratch/memoozd/gadplus/runs/lj-lambda2-compare-${SLURM_JOB_ID}}
N_SAMPLES=48
NOISES=0.10,0.20
N_TASKS=192
export RUN_ROOT N_SAMPLES NOISES

mkdir -p "$RUN_ROOT" /scratch/memoozd/gadplus/logs /scratch/memoozd/gadplus/tmp
cd "$WORK_ROOT"
source .venv/bin/activate
export PYTHONPATH="$WORK_ROOT/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/scratch/memoozd/gadplus/tmp
export PYTHONUNBUFFERED=1

python scripts/lj_intrinsic_noise_sweep.py prepare --output-root "$RUN_ROOT"
srun --exclusive --exact --nodes=1 --ntasks="$N_TASKS" --cpus-per-task=1 --cpu-bind=cores \
    bash -c 'python scripts/lj_lambda2_noise_comparison.py worker \
        --output-root "$RUN_ROOT" --task-id "$SLURM_PROCID" \
        --n-samples "$N_SAMPLES" --noises "$NOISES"'
python scripts/lj_lambda2_noise_comparison.py aggregate \
    --output-root "$RUN_ROOT" --expected-tasks "$N_TASKS"
echo "run_root=$RUN_ROOT"
