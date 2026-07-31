#!/bin/bash
#SBATCH --job-name=t1x_gad_287_raw
#SBATCH --account=def-aspuru
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/t1x_gad_287_raw_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/t1x_gad_287_raw_%j.err

# Matched ordinary projected-GAD arm for the 0.20 A test-287 comparison.
# There is no descent gate, spectral blend, preconditioner, adaptive timestep,
# or mode-history state.  The seed schedule matches the competitive and Sella
# campaigns exactly.
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5
PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
OUT_ROOT=/scratch/memoozd/gadplus/runs/t1x-gxtb-regular-gad-287-raw-${SLURM_JOB_ID}
export PROJECT_DIR OUT_ROOT
export GADPLUS_SCRATCH=/scratch/memoozd/gadplus
export GADPLUS_T1X_H5=/scratch/memoozd/GAD/data/transition1x.h5
export GADPLUS_GXTB_EXE=/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb
export GADPLUS_GXTB_PARALLEL=32 GADPLUS_ENV="$PROJECT_DIR/.venv"
export GADPLUS_GXTB_WORK_ROOT=/scratch/memoozd/gadplus/tmp/gxtb_regular_gad_287_${SLURM_JOB_ID}
source "$PROJECT_DIR/scripts/lib/gxtb_runtime.sh"
export PYTHON PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"
run_task() {
  local task=$1 base_seed=$((20260727 + task))
  "$PYTHON" -u "$PROJECT_DIR/scripts/gad_smoke.py" \
    --backend gxtb --method gxTB --sample-indices "$task" \
    --start-geometry noised_ts --noise 0.20 --seed "$base_seed" \
    --n-steps 300 --dt 0.003 --force-threshold 0.03 --force-criterion fmax \
    --max-atom-disp 0.35 --n-workers 1 \
    --output-dir "$OUT_ROOT/task_$(printf '%03d' "$task")"
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
