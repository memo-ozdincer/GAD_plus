#!/bin/bash
#SBATCH --job-name=lj_intrinsic_noise
#SBATCH --account=def-aspuru
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/lj_intrinsic_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/lj_intrinsic_%j.err

# Full-node packed analytic-LJ sweep. One exclusive Slurm step launches 192
# independent one-core tasks, one per shard; there is no nested Python process
# pool and no trajectory oversubscription. All code, environments, logs, and
# outputs remain under the required Trillium scratch roots.

set -euo pipefail

module purge
module load StdEnv/2023 python/3.11.5

WORK_ROOT=/scratch/memoozd/GAD/GAD_plus
RUN_ROOT=${RUN_ROOT:-/scratch/memoozd/gadplus/runs/lj-intrinsic-noise-${SLURM_JOB_ID}}
N_SHARDS=192
N_SAMPLES=${N_SAMPLES:-96}
NOISES=${NOISES:-0,0.005,0.01,0.02,0.03,0.05,0.075,0.10,0.125,0.15,0.20,0.25,0.30,0.40,0.50}
# Colons are accepted for sbatch --export, whose own delimiter is a comma.
NOISES=${NOISES//:/,}
MAX_STEPS=${MAX_STEPS:-200}
export RUN_ROOT N_SHARDS N_SAMPLES NOISES MAX_STEPS

mkdir -p "$RUN_ROOT" /scratch/memoozd/gadplus/logs /scratch/memoozd/gadplus/tmp
cd "$WORK_ROOT"
source .venv/bin/activate

export PYTHONPATH="$WORK_ROOT/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export TMPDIR=/scratch/memoozd/gadplus/tmp
export PYTHONUNBUFFERED=1

python scripts/lj_intrinsic_noise_sweep.py prepare --output-root "$RUN_ROOT"

srun --exclusive --exact --nodes=1 --ntasks="$N_SHARDS" --cpus-per-task=1 --cpu-bind=cores \
    bash -c 'python scripts/lj_intrinsic_noise_sweep.py worker \
        --output-root "$RUN_ROOT" \
        --shard-id "$SLURM_PROCID" \
        --n-shards "$N_SHARDS" \
        --n-samples "$N_SAMPLES" \
        --noises "$NOISES" \
        --max-steps "$MAX_STEPS"'

python scripts/lj_intrinsic_noise_sweep.py aggregate \
    --output-root "$RUN_ROOT" \
    --expected-shards "$N_SHARDS"

echo "run_root=$RUN_ROOT"
