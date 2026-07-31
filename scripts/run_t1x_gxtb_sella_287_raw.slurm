#!/bin/bash
#SBATCH --job-name=t1x_sella_287_raw
#SBATCH --account=def-aspuru
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/t1x_sella_287_raw_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/t1x_sella_287_raw_%j.err

# Paired Sella arm of the fixed test-287, 0.20 A campaign. Each task receives
# the identical noise tensor used by competitive intrinsic GAD.
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5
PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
OUT_ROOT=/scratch/memoozd/gadplus/runs/t1x-gxtb-sella-287-raw-${SLURM_JOB_ID}
export PROJECT_DIR OUT_ROOT
export GADPLUS_SCRATCH=/scratch/memoozd/gadplus
export GADPLUS_T1X_H5=/scratch/memoozd/GAD/data/transition1x.h5
export GADPLUS_GXTB_EXE=/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb
export GADPLUS_GXTB_PARALLEL=32 GADPLUS_ENV="$PROJECT_DIR/.venv"
export GADPLUS_GXTB_WORK_ROOT=/scratch/memoozd/gadplus/tmp/gxtb_sella_287_${SLURM_JOB_ID}
source "$PROJECT_DIR/scripts/lib/gxtb_runtime.sh"
export PYTHON PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"
run_task() {
  local task=$1 base_seed=$((20260727 + task))
  "$PYTHON" -u "$PROJECT_DIR/scripts/sella_smoke.py" \
    --backend gxtb --method gxTB --sample-indices "$task" \
    --start-geometry noised_ts --noise 0.20 --seed "$base_seed" \
    --max-steps 300 --fmax 0.03 --apply-eckart --delta0 0.1 --gamma 0.4 \
    --diag-every 1 --n-workers 1 --output-dir "$OUT_ROOT/task_$(printf '%03d' "$task")"
}
export -f run_task
run_wave() {
  local offset=$1 count=$2
  srun --ntasks="$count" --cpus-per-task=32 --cpu-bind=cores \
    bash -c 'run_task $((SLURM_PROCID + '"$offset"'))'
}
for offset in 0 48 96 144 192; do run_wave "$offset" 48; done
run_wave 240 47
echo "run_root=$OUT_ROOT"
