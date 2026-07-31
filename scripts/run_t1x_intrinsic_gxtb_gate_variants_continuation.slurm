#!/bin/bash
#SBATCH --job-name=t1x_gate_continue
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/t1x_gate_continue_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/t1x_gate_continue_%j.err
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5

PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
PILOT_DIR=/scratch/memoozd/gadplus/tmp/gadplus-push.ZiJ3Oe
SOURCE_ROOT=/scratch/memoozd/gadplus/runs/t1x-intrinsic-gxtb-ablation-1950894/eta001_tau001
RUN_ROOT=/scratch/memoozd/gadplus/runs/t1x-intrinsic-gxtb-gate-continuation-${SLURM_JOB_ID}
export GADPLUS_SCRATCH=/scratch/memoozd/gadplus
export GADPLUS_T1X_H5=/scratch/memoozd/GAD/data/transition1x.h5
export GADPLUS_GXTB_EXE=/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb
export GADPLUS_GXTB_PARALLEL=32 GADPLUS_ENV="$PROJECT_DIR/.venv"
export GADPLUS_GXTB_WORK_ROOT=/scratch/memoozd/gadplus/tmp/gxtb_gate_continue_${SLURM_JOB_ID}
source "$PROJECT_DIR/scripts/lib/gxtb_runtime.sh"
export PYTHON PYTHONPATH="$PROJECT_DIR/src:$PILOT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$RUN_ROOT"

shell_wave() {
  local variant=$1 offset=$2 out=$3
  srun --ntasks=6 --ntasks-per-node=6 --cpus-per-task=32 --cpu-bind=cores \
    bash -c 'task=$((SLURM_PROCID + '"$offset"')); "$PYTHON" '"$PROJECT_DIR"'/scripts/t1x_intrinsic_gxtb_minimum_shell.py --source-summary '"$SOURCE_ROOT"'/summary.json --source-root '"$SOURCE_ROOT"' --output-root '"$out"' --task-id "$task" --gxtb-executable "$GADPLUS_GXTB_EXE" --parallel "$GADPLUS_GXTB_PARALLEL" --gate-variant '"$variant"
}
raw_wave() {
  local offset=$1 out=$2
  srun --ntasks=6 --ntasks-per-node=6 --cpus-per-task=32 --cpu-bind=cores \
    bash -c 'task=$((SLURM_PROCID + '"$offset"')); "$PYTHON" '"$PILOT_DIR"'/scripts/t1x_intrinsic_gxtb_pilot.py worker --output-root '"$out"' --task-id "$task" --gxtb-executable "$GADPLUS_GXTB_EXE" --parallel "$GADPLUS_GXTB_PARALLEL" --max-steps 300 --step-fraction 0.01 --spectral-temperature 0.01 --gate-variant competitive'
}

# Finish the twelve missing alignment shell branches (task ids 12--23).
for offset in 12 18; do shell_wave alignment "$offset" "$RUN_ROOT/alignment_shell_remainder"; done

raw="$RUN_ROOT/competitive_raw"
mkdir -p "$raw" "$RUN_ROOT/competitive_shell"
ln -s "$SOURCE_ROOT/native_labels" "$raw/native_labels"
raw_wave 0 "$raw"
raw_wave 6 "$raw"
"$PYTHON" "$PILOT_DIR/scripts/t1x_intrinsic_gxtb_pilot.py" aggregate \
  --output-root "$raw" --expected-tasks 12 --gxtb-executable "$GADPLUS_GXTB_EXE" \
  --parallel "$GADPLUS_GXTB_PARALLEL" --max-steps 300 --step-fraction 0.01 \
  --spectral-temperature 0.01 --gate-variant competitive
for offset in 0 6 12 18; do shell_wave competitive "$offset" "$RUN_ROOT/competitive_shell"; done
echo "run_root=$RUN_ROOT"
