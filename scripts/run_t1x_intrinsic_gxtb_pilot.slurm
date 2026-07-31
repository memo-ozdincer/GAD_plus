#!/bin/bash
#SBATCH --job-name=t1x_intrinsic_gxtb
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/t1x_intrinsic_gxtb_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/t1x_intrinsic_gxtb_%j.err

# Three T1x molecules, each from the labelled structure and three Cartesian
# noise levels.  Six independent 32-core numerical-Hessian tasks fill the
# 192-core node.  The second search wave supplies task IDs 6--11.
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5

PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
PILOT_DIR=/scratch/memoozd/gadplus/tmp/gadplus-push.ZiJ3Oe
export GADPLUS_SCRATCH=/scratch/memoozd/gadplus
export GADPLUS_T1X_H5=/scratch/memoozd/GAD/data/transition1x.h5
export GADPLUS_GXTB_EXE=/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb
export GADPLUS_GXTB_PARALLEL=32
export GADPLUS_ENV="$PROJECT_DIR/.venv"
export GADPLUS_GXTB_WORK_ROOT=/scratch/memoozd/gadplus/tmp/gxtb_t1x_intrinsic_${SLURM_JOB_ID}
OUT_ROOT=/scratch/memoozd/gadplus/runs/t1x-intrinsic-gxtb-${SLURM_JOB_ID}
export OUT_ROOT

source "$PROJECT_DIR/scripts/lib/gxtb_runtime.sh"
export PYTHON
export PYTHONPATH="$PROJECT_DIR/src:$PILOT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"

run_tasks() {
  local command=$1
  local offset=$2
  srun --ntasks=6 --ntasks-per-node=6 --cpus-per-task=32 --cpu-bind=cores \
    bash -c 'task=$((SLURM_PROCID + '"$offset"')); "$PYTHON" '"$PILOT_DIR"'/scripts/t1x_intrinsic_gxtb_pilot.py '"$command"' --output-root "$OUT_ROOT" --task-id "$task" --gxtb-executable "$GADPLUS_GXTB_EXE" --parallel "$GADPLUS_GXTB_PARALLEL"'
}

run_tasks prepare-native 0
run_tasks worker 0
run_tasks worker 6
"$PYTHON" "$PILOT_DIR/scripts/t1x_intrinsic_gxtb_pilot.py" aggregate \
  --output-root "$OUT_ROOT" --expected-tasks 12 \
  --gxtb-executable "$GADPLUS_GXTB_EXE" --parallel "$GADPLUS_GXTB_PARALLEL"
echo "run_root=$OUT_ROOT"
