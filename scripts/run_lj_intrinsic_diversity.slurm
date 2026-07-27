#!/bin/bash
#SBATCH --job-name=lj_intrinsic_diversity
#SBATCH --account=def-aspuru
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/lj_diversity_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/lj_diversity_%j.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5

WORK_ROOT=/scratch/memoozd/GAD/GAD_plus
RUN_ROOT=${RUN_ROOT:-/scratch/memoozd/gadplus/runs/lj-intrinsic-noise-1940584}
N_SHARDS=192
export RUN_ROOT N_SHARDS

cd "$WORK_ROOT"
source .venv/bin/activate
export PYTHONPATH="$WORK_ROOT/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/scratch/memoozd/gadplus/tmp
export PYTHONUNBUFFERED=1

srun --exclusive --exact --nodes=1 --ntasks="$N_SHARDS" --cpus-per-task=1 --cpu-bind=cores \
    bash -c 'python scripts/analyze_lj_intrinsic_diversity.py worker \
        --run-root "$RUN_ROOT" --shard-id "$SLURM_PROCID" --n-shards "$N_SHARDS"'
python scripts/analyze_lj_intrinsic_diversity.py aggregate \
    --run-root "$RUN_ROOT" --expected-shards "$N_SHARDS"
