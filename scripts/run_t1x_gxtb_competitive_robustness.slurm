#!/bin/bash
#SBATCH --job-name=t1x_comp_robust
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/t1x_comp_robust_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/t1x_comp_robust_%j.err

# Predeclared development-only robustness check for competitive intrinsic GAD.
# Grid: eta,tau in {0.005,0.01,0.02}; 12 reactions x {0.10,0.20 A} starts.
# Do not use the test-287 campaigns to choose a cell.  After completion select
# the central (eta,tau)=(0.01,0.01) point whenever it lies within one intended
# topology recovery of the best cell at *each* noise level; otherwise choose
# the nearest grid point satisfying that plateau rule.
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5
PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
PILOT_DIR=/scratch/memoozd/gadplus/tmp/gadplus-push.ZiJ3Oe
LABEL_ROOT=/scratch/memoozd/gadplus/runs/t1x-intrinsic-gxtb-heldout-gates-1953766/native_labels
SAMPLE_IDS=50,92,134,176,178,180,184,185,201,243,264,285
RUN_ROOT=/scratch/memoozd/gadplus/runs/t1x-intrinsic-gxtb-competitive-robustness-${SLURM_JOB_ID}
export GADPLUS_SCRATCH=/scratch/memoozd/gadplus
export GADPLUS_T1X_H5=/scratch/memoozd/GAD/data/transition1x.h5
export GADPLUS_GXTB_EXE=/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb
export GADPLUS_GXTB_PARALLEL=32 GADPLUS_ENV="$PROJECT_DIR/.venv"
export GADPLUS_GXTB_WORK_ROOT=/scratch/memoozd/gadplus/tmp/gxtb_comp_robust_${SLURM_JOB_ID}
source "$PROJECT_DIR/scripts/lib/gxtb_runtime.sh"
export PYTHON PYTHONPATH="$PROJECT_DIR/src:$PILOT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$RUN_ROOT"
run_wave() {
  local out=$1 eta=$2 tau=$3 offset=$4
  srun --ntasks=6 --ntasks-per-node=6 --cpus-per-task=32 --cpu-bind=cores \
    bash -c 'task=$((SLURM_PROCID + '"$offset"')); "$PYTHON" '"$PILOT_DIR"'/scripts/t1x_intrinsic_gxtb_pilot.py worker --output-root '"$out"' --task-id "$task" --sample-ids '"$SAMPLE_IDS"' --noise-levels 0.10,0.20 --gxtb-executable "$GADPLUS_GXTB_EXE" --parallel "$GADPLUS_GXTB_PARALLEL" --max-steps 300 --step-fraction '"$eta"' --spectral-temperature '"$tau"' --gate-variant competitive'
}
for spec in eta0005_tau0005:0.005:0.005 eta0005_tau001:0.005:0.01 eta0005_tau002:0.005:0.02 eta001_tau0005:0.01:0.005 eta001_tau001:0.01:0.01 eta001_tau002:0.01:0.02 eta002_tau0005:0.02:0.005 eta002_tau001:0.02:0.01 eta002_tau002:0.02:0.02; do
  IFS=: read -r label eta tau <<< "$spec"
  out="$RUN_ROOT/$label"
  mkdir -p "$out"
  ln -s "$LABEL_ROOT" "$out/native_labels"
  for offset in 0 6 12 18; do run_wave "$out" "$eta" "$tau" "$offset"; done
  "$PYTHON" "$PILOT_DIR/scripts/t1x_intrinsic_gxtb_pilot.py" aggregate \
    --output-root "$out" --expected-tasks 24 --sample-ids "$SAMPLE_IDS" \
    --noise-levels 0.10,0.20 --gxtb-executable "$GADPLUS_GXTB_EXE" \
    --parallel "$GADPLUS_GXTB_PARALLEL" --max-steps 300 --step-fraction "$eta" \
    --spectral-temperature "$tau" --gate-variant competitive
done
echo "run_root=$RUN_ROOT"
