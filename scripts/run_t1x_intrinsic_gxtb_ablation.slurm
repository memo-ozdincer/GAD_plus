#!/bin/bash
#SBATCH --job-name=t1x_intrinsic_ablate
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/t1x_intrinsic_ablate_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/t1x_intrinsic_ablate_%j.err

# A 3 x 2 factorial: step fraction {0.05, 0.02, 0.01} x spectral
# temperature {0.01, 0.03}.  Every condition uses the identical twelve fixed
# T1x starts and 300 pointwise updates.  Six independent 32-core numerical
# Hessian tasks fill the 192-core CPU node in every Slurm step.
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
export GADPLUS_GXTB_WORK_ROOT=/scratch/memoozd/gadplus/tmp/gxtb_t1x_ablation_${SLURM_JOB_ID}
RUN_ROOT=/scratch/memoozd/gadplus/runs/t1x-intrinsic-gxtb-ablation-${SLURM_JOB_ID}

source "$PROJECT_DIR/scripts/lib/gxtb_runtime.sh"
export PYTHON
export PYTHONPATH="$PROJECT_DIR/src:$PILOT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$RUN_ROOT"

run_wave() {
  local command=$1 offset=$2 output_root=$3 eta=$4 tau=$5
  srun --ntasks=6 --ntasks-per-node=6 --cpus-per-task=32 --cpu-bind=cores \
    bash -c 'task=$((SLURM_PROCID + '"$offset"')); "$PYTHON" '"$PILOT_DIR"'/scripts/t1x_intrinsic_gxtb_pilot.py '"$command"' --output-root '"$output_root"' --task-id "$task" --gxtb-executable "$GADPLUS_GXTB_EXE" --parallel "$GADPLUS_GXTB_PARALLEL" --max-steps 300 --step-fraction '"$eta"' --spectral-temperature '"$tau"
}

for spec in eta005_tau001:0.05:0.01 eta005_tau003:0.05:0.03 eta002_tau001:0.02:0.01 eta002_tau003:0.02:0.03 eta001_tau001:0.01:0.01 eta001_tau003:0.01:0.03; do
  IFS=: read -r label eta tau <<< "$spec"
  out="$RUN_ROOT/$label"
  mkdir -p "$out"
  run_wave prepare-native 0 "$out" "$eta" "$tau"
  run_wave worker 0 "$out" "$eta" "$tau"
  run_wave worker 6 "$out" "$eta" "$tau"
  "$PYTHON" "$PILOT_DIR/scripts/t1x_intrinsic_gxtb_pilot.py" aggregate \
    --output-root "$out" --expected-tasks 12 \
    --gxtb-executable "$GADPLUS_GXTB_EXE" --parallel "$GADPLUS_GXTB_PARALLEL" \
    --max-steps 300 --step-fraction "$eta" --spectral-temperature "$tau"
done
echo "run_root=$RUN_ROOT"
