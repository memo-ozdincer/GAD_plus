#!/bin/bash
#SBATCH --job-name=lj_multisize_cs2
#SBATCH --account=def-aspuru
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/lj_multisize_cs2_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/lj_multisize_cs2_%j.err

# Full-node packed analytic-LJ campaign. Every Slurm step is one independent
# trajectory; no trajectory creates a nested process pool.
set -euo pipefail

module purge
module load StdEnv/2023 python/3.11.5

WORK_ROOT=/scratch/memoozd/GAD/GAD_plus
RUN_ROOT=${RUN_ROOT:-/scratch/memoozd/gadplus/runs/lj-multisize-cs2-${SLURM_JOB_ID}}
SIZES=${SIZES:-13,31,38,55,75}
METHODS=${METHODS:-intrinsic_lambda2,cs2,sella}
START_FAMILIES=${START_FAMILIES:-mode_push,cartesian_noise}
LEVELS=${LEVELS:-0.10,0.20}
N_SAMPLES=${N_SAMPLES:-16}
MAX_STEPS=${MAX_STEPS:-500}

IFS=',' read -r -a SIZE_ARRAY <<< "$SIZES"
IFS=',' read -r -a METHOD_ARRAY <<< "$METHODS"
IFS=',' read -r -a FAMILY_ARRAY <<< "$START_FAMILIES"
IFS=',' read -r -a LEVEL_ARRAY <<< "$LEVELS"
N_BASINS=${#SIZE_ARRAY[@]}
[[ ",$SIZES," == *,38,* ]] && N_BASINS=$((N_BASINS + 1))
[[ ",$SIZES," == *,75,* ]] && N_BASINS=$((N_BASINS + 1))
N_TASKS=$((${#METHOD_ARRAY[@]} * N_BASINS * ${#FAMILY_ARRAY[@]} * ${#LEVEL_ARRAY[@]} * N_SAMPLES))

mkdir -p "$RUN_ROOT" /scratch/memoozd/gadplus/logs /scratch/memoozd/gadplus/tmp \
    "/scratch/memoozd/gadplus/cache/lj_multisize_${SLURM_JOB_ID}"
cd "$WORK_ROOT"
source .venv/bin/activate
export PYTHONPATH="$WORK_ROOT/src:$WORK_ROOT/scripts"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/scratch/memoozd/gadplus/tmp PYTHONUNBUFFERED=1

COMMON=(--output-root "$RUN_ROOT" --sizes "$SIZES" --methods "$METHODS"
        --start-families "$START_FAMILIES" --levels "$LEVELS"
        --n-samples "$N_SAMPLES" --max-steps "$MAX_STEPS")
python scripts/lj_multisize_cs2_benchmark.py prepare "${COMMON[@]}"

export RUN_ROOT SIZES METHODS START_FAMILIES LEVELS N_SAMPLES MAX_STEPS
active=0
for ((task_id=0; task_id<N_TASKS; task_id++)); do
    task_path=$(printf '%s/tasks/task_%05d.json' "$RUN_ROOT" "$task_id")
    if [[ -s "$task_path" ]] && jq -e '.calculator_valid == true' "$task_path" >/dev/null; then
        continue
    fi
    export TASK_ID="$task_id"
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --cpu-bind=cores \
        --mem=3G \
        bash -c 'export XDG_CACHE_HOME="/scratch/memoozd/gadplus/cache/lj_multisize_${SLURM_JOB_ID}/task_${TASK_ID}"; \
          python scripts/lj_multisize_cs2_benchmark.py worker \
          --output-root "$RUN_ROOT" --task-id "$TASK_ID" \
          --sizes "$SIZES" --methods "$METHODS" --start-families "$START_FAMILIES" \
          --levels "$LEVELS" --n-samples "$N_SAMPLES" --max-steps "$MAX_STEPS"' &
    active=$((active + 1))
    if (( active >= 192 )); then
        wait -n
        active=$((active - 1))
    fi
done
wait

python scripts/lj_multisize_cs2_benchmark.py aggregate "${COMMON[@]}" --expected-tasks "$N_TASKS"
python scripts/summarize_lj_multisize_results.py "$RUN_ROOT" --expected-tasks "$N_TASKS"
echo "run_root=$RUN_ROOT"
