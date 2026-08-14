#!/bin/bash
#SBATCH --job-name=lj_gad_rescue
#SBATCH --account=def-aspuru
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/lj_gad_rescue_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/lj_gad_rescue_%j.err

# Exactly 192 independent deterministic rescue trajectories fill the CPU node.
set -euo pipefail

module purge
module load StdEnv/2023 python/3.11.5

WORK_ROOT=/scratch/memoozd/GAD/GAD_plus
PARENT_ROOT=/scratch/memoozd/gadplus/runs/lj-multisize-cs2-2076554
RUN_ROOT=${RUN_ROOT:-/scratch/memoozd/gadplus/runs/lj-gad-rescue-${SLURM_JOB_ID}}

mkdir -p "$RUN_ROOT" /scratch/memoozd/gadplus/logs /scratch/memoozd/gadplus/tmp \
  "/scratch/memoozd/gadplus/cache/lj_rescue_${SLURM_JOB_ID}"
cd "$WORK_ROOT"
source .venv/bin/activate
export PYTHONPATH="$WORK_ROOT/src:$WORK_ROOT/scripts"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/scratch/memoozd/gadplus/tmp PYTHONUNBUFFERED=1

python scripts/lj_gad_rescue_benchmark.py prepare \
  --output-root "$RUN_ROOT" --parent-root "$PARENT_ROOT"

for ((task_id=0; task_id<192; task_id++)); do
  task_path=$(printf '%s/tasks/task_%03d.json' "$RUN_ROOT" "$task_id")
  if [[ -s "$task_path" ]] && jq -e '.calculator_valid == true' "$task_path" >/dev/null; then
    continue
  fi
  export TASK_ID=$task_id
  # Without a per-step memory request, this Slurm version gives the first
  # exclusive step the allocation's entire --mem=0 node and serializes all
  # later steps.  Three GiB per analytic-LJ worker keeps 192 steps below the
  # 767 GiB node allocation while preserving one core per trajectory.
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --mem=3000M --cpu-bind=cores \
    bash -c 'export XDG_CACHE_HOME="/scratch/memoozd/gadplus/cache/lj_rescue_${SLURM_JOB_ID}/task_${TASK_ID}"; \
      python scripts/lj_gad_rescue_benchmark.py worker \
      --output-root "'$RUN_ROOT'" --parent-root "'$PARENT_ROOT'" --task-id "$TASK_ID"' &
done
wait

python scripts/lj_gad_rescue_benchmark.py aggregate \
  --output-root "$RUN_ROOT" --parent-root "$PARENT_ROOT" --expected-tasks 192
echo "run_root=$RUN_ROOT"
