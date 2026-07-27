#!/bin/bash
#SBATCH --job-name=lj_method_progression
#SBATCH --account=def-aspuru
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/lj_method_progression_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/lj_method_progression_%j.err

# Full-node packed analytic-LJ comparison. Each task is one independent
# trajectory/method pair and uses a single CPU; no task starts a nested pool.
set -euo pipefail

module purge
module load StdEnv/2023 python/3.11.5

WORK_ROOT=/scratch/memoozd/GAD/GAD_plus
RUN_ROOT=${RUN_ROOT:-/scratch/memoozd/gadplus/runs/lj-method-progression-${SLURM_JOB_ID}}
N_SAMPLES=${N_SAMPLES:-48}
NOISES=${NOISES:-0.10,0.20,0.40}
METHODS=${METHODS:-ordinary_gad,hard_gate,historical_lambda2,intrinsic}
IFS=',' read -r -a METHOD_ARRAY <<< "$METHODS"
IFS=',' read -r -a NOISE_ARRAY <<< "$NOISES"
N_TASKS=$((${#METHOD_ARRAY[@]} * 2 * ${#NOISE_ARRAY[@]} * N_SAMPLES))
TASKS_PER_WAVE=192
export RUN_ROOT N_SAMPLES NOISES METHODS N_TASKS

mkdir -p "$RUN_ROOT" /scratch/memoozd/gadplus/logs /scratch/memoozd/gadplus/tmp
cd "$WORK_ROOT"
source .venv/bin/activate
export PYTHONPATH="$WORK_ROOT/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1 TMPDIR=/scratch/memoozd/gadplus/tmp PYTHONUNBUFFERED=1

python scripts/lj_method_progression_sweep.py prepare --output-root "$RUN_ROOT"

for ((offset=0; offset<N_TASKS; offset+=TASKS_PER_WAVE)); do
    export TASK_OFFSET="$offset"
    remaining=$((N_TASKS - offset))
    count=$(( remaining < TASKS_PER_WAVE ? remaining : TASKS_PER_WAVE ))
    srun --exclusive --exact --nodes=1 --ntasks="$count" --cpus-per-task=1 --cpu-bind=cores \
        bash -c 'python scripts/lj_method_progression_sweep.py worker \
            --output-root "$RUN_ROOT" --task-id "$((TASK_OFFSET + SLURM_PROCID))" \
            --n-samples "$N_SAMPLES" --noises "$NOISES" --methods "$METHODS"'
done

python scripts/lj_method_progression_sweep.py aggregate \
    --output-root "$RUN_ROOT" --expected-tasks "$N_TASKS"
echo "run_root=$RUN_ROOT"
