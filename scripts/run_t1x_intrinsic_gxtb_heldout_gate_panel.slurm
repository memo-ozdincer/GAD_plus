#!/bin/bash
#SBATCH --job-name=t1x_heldout_gates
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/t1x_heldout_gates_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/t1x_heldout_gates_%j.err

# Held-out Transition1x reactions, paired original-vs-competitive gates.
# At each gate, 12 reactions x {0.10, 0.20 A} use identical seeded starts.
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5
PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
PILOT_DIR=/scratch/memoozd/gadplus/tmp/gadplus-push.ZiJ3Oe
SAMPLE_IDS=50,92,134,176,178,180,184,185,201,243,264,285
RUN_ROOT=/scratch/memoozd/gadplus/runs/t1x-intrinsic-gxtb-heldout-gates-${SLURM_JOB_ID}
export GADPLUS_SCRATCH=/scratch/memoozd/gadplus
export GADPLUS_T1X_H5=/scratch/memoozd/GAD/data/transition1x.h5
export GADPLUS_GXTB_EXE=/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb
export GADPLUS_GXTB_PARALLEL=32 GADPLUS_ENV="$PROJECT_DIR/.venv"
export GADPLUS_GXTB_WORK_ROOT=/scratch/memoozd/gadplus/tmp/gxtb_heldout_gates_${SLURM_JOB_ID}
source "$PROJECT_DIR/scripts/lib/gxtb_runtime.sh"
export PYTHON PYTHONPATH="$PROJECT_DIR/src:$PILOT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$RUN_ROOT/native_labels"

prepare_wave() {
  local offset=$1
  srun --ntasks=6 --ntasks-per-node=6 --cpus-per-task=32 --cpu-bind=cores \
    bash -c 'task=$((SLURM_PROCID + '"$offset"')); "$PYTHON" '"$PILOT_DIR"'/scripts/t1x_intrinsic_gxtb_pilot.py prepare-native --output-root '"$RUN_ROOT"' --task-id "$task" --sample-ids '"$SAMPLE_IDS"' --gxtb-executable "$GADPLUS_GXTB_EXE" --parallel "$GADPLUS_GXTB_PARALLEL"'
}
worker_wave() {
  local variant=$1 offset=$2 out=$3
  srun --ntasks=6 --ntasks-per-node=6 --cpus-per-task=32 --cpu-bind=cores \
    bash -c 'task=$((SLURM_PROCID + '"$offset"')); "$PYTHON" '"$PILOT_DIR"'/scripts/t1x_intrinsic_gxtb_pilot.py worker --output-root '"$out"' --task-id "$task" --sample-ids '"$SAMPLE_IDS"' --noise-levels 0.10,0.20 --gxtb-executable "$GADPLUS_GXTB_EXE" --parallel "$GADPLUS_GXTB_PARALLEL" --max-steps 300 --step-fraction 0.01 --spectral-temperature 0.01 --gate-variant '"$variant"
}

for offset in 0 6 12 18; do prepare_wave "$offset"; done
for variant in lambda2 competitive; do
  out="$RUN_ROOT/$variant"
  mkdir -p "$out"
  ln -s "$RUN_ROOT/native_labels" "$out/native_labels"
  for offset in 0 6 12 18; do worker_wave "$variant" "$offset" "$out"; done
  "$PYTHON" "$PILOT_DIR/scripts/t1x_intrinsic_gxtb_pilot.py" aggregate \
    --output-root "$out" --expected-tasks 24 --sample-ids "$SAMPLE_IDS" \
    --noise-levels 0.10,0.20 --gxtb-executable "$GADPLUS_GXTB_EXE" \
    --parallel "$GADPLUS_GXTB_PARALLEL" --max-steps 300 --step-fraction 0.01 \
    --spectral-temperature 0.01 --gate-variant "$variant"
done
echo "run_root=$RUN_ROOT"
