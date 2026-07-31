#!/bin/bash
#SBATCH --job-name=t1x_gxtb_sella
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/t1x_gxtb_sella_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/t1x_gxtb_sella_%j.err

# Fair Sella comparator: same 12 held-out IDs, exact same noise tensors as
# the intrinsic pilot, g-xTB full Hessian every Sella step, Cartesian+Eckart.
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5
PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
OUT_ROOT=/scratch/memoozd/gadplus/runs/t1x-gxtb-sella-heldout-${SLURM_JOB_ID}
export PROJECT_DIR OUT_ROOT
export GADPLUS_SCRATCH=/scratch/memoozd/gadplus
export GADPLUS_T1X_H5=/scratch/memoozd/GAD/data/transition1x.h5
export GADPLUS_GXTB_EXE=/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb
export GADPLUS_GXTB_PARALLEL=32 GADPLUS_ENV="$PROJECT_DIR/.venv"
export GADPLUS_GXTB_WORK_ROOT=/scratch/memoozd/gadplus/tmp/gxtb_sella_heldout_${SLURM_JOB_ID}
source "$PROJECT_DIR/scripts/lib/gxtb_runtime.sh"
export PYTHON PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"

run_task() {
  local task=$1
  local samples=(50 92 134 176 178 180 184 185 201 243 264 285)
  local sample_index=$((task / 2))
  local noise_index=$((task % 2))
  local sample=${samples[$sample_index]}
  local noises=(0.10 0.20)
  local noise=${noises[$noise_index]}
  local base_seed=$((20260727 + task))
  "$PYTHON" -u "$PROJECT_DIR/scripts/sella_smoke.py" \
    --backend gxtb --method gxTB --sample-indices "$sample" \
    --start-geometry noised_ts --noise "$noise" --seed "$base_seed" \
    --max-steps 300 --fmax 0.03 --apply-eckart --delta0 0.1 --gamma 0.4 \
    --diag-every 1 --n-workers 1 --output-dir "$OUT_ROOT/task_$(printf '%03d' "$task")"
}
export -f run_task
for offset in 0 6 12 18; do
  srun --ntasks=6 --ntasks-per-node=6 --cpus-per-task=32 --cpu-bind=cores \
    bash -c 'run_task $((SLURM_PROCID + '"$offset"'))'
done
echo "run_root=$OUT_ROOT"
