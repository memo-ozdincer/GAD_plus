#!/bin/bash
#SBATCH --job-name=t1x_subspace_287
#SBATCH --account=def-aspuru
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/t1x_subspace_287_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/t1x_subspace_287_%j.err

# Full fixed test-287 validation of the experimental relative-soft-subspace
# reflection.  This is paired with the frozen competitive 0.20-A campaign:
# each sample id reproduces its original noised TS exactly.  The step map is
# otherwise local and unchanged; endpoints are intentionally a separate pass.
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5

PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
export SAMPLE_IDS=$(seq -s, 0 286)
OUT_ROOT=/scratch/memoozd/gadplus/runs/t1x-gxtb-competitive-subspace-287-raw-${SLURM_JOB_ID}
export GADPLUS_SCRATCH=/scratch/memoozd/gadplus
export GADPLUS_T1X_H5=/scratch/memoozd/GAD/data/transition1x.h5
export GADPLUS_GXTB_EXE=/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb
export GADPLUS_GXTB_PARALLEL=32 GADPLUS_ENV="$PROJECT_DIR/.venv"
export GADPLUS_GXTB_WORK_ROOT=/scratch/memoozd/gadplus/tmp/gxtb_subspace_287_${SLURM_JOB_ID}
source "$PROJECT_DIR/scripts/lib/gxtb_runtime.sh"
export PYTHON PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"

run_wave() {
  local offset=$1 count=$2
  srun --ntasks="$count" --cpus-per-task=32 --cpu-bind=cores \
    bash -c 'task=$((SLURM_PROCID + '"$offset"')); "$PYTHON" '"$PROJECT_DIR"'/scripts/t1x_intrinsic_gxtb_pilot.py worker --output-root '"$OUT_ROOT"' --task-id "$task" --sample-ids "$SAMPLE_IDS" --noise-levels 0.20 --skip-endpoints --gxtb-executable "$GADPLUS_GXTB_EXE" --parallel "$GADPLUS_GXTB_PARALLEL" --max-steps 300 --step-fraction 0.01 --spectral-temperature 0.01 --gate-variant competitive_subspace --record-trajectories --campaign t1x-gxtb-competitive-subspace-287-0p20 --selection-stage evaluation'
}

for offset in 0 48 96 144 192; do run_wave "$offset" 48; done
run_wave 240 47
"$PYTHON" "$PROJECT_DIR/scripts/t1x_intrinsic_gxtb_pilot.py" aggregate \
  --output-root "$OUT_ROOT" --expected-tasks 287 --sample-ids "$SAMPLE_IDS" \
  --noise-levels 0.20
echo "run_root=$OUT_ROOT"
