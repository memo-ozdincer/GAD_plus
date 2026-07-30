#!/bin/bash
#SBATCH --job-name=t1x_intrinsic_topo
#SBATCH --account=def-aspuru
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/t1x_intrinsic_topo_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/t1x_intrinsic_topo_%j.err

# Native endpoint scorer for any completed 287-start intrinsic-GAD raw root.
# SOURCE_ROOT is required and contains task JSON records; every potential
# candidate gets a deterministic task slot, while noncandidates return a
# padding record without calculator work.  The scorer is observational and
# never changes the local search outcome.
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5

PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
: "${SOURCE_ROOT:?submit with SOURCE_ROOT=/scratch/.../raw-root}"
LABEL_ROOT=/scratch/memoozd/gadplus/runs/t1x-gxtb-native-labels-287-1961128/native_labels
OUT_ROOT=/scratch/memoozd/gadplus/runs/t1x-gxtb-intrinsic-topology-$(basename "$SOURCE_ROOT")-${SLURM_JOB_ID}
export GADPLUS_SCRATCH=/scratch/memoozd/gadplus
export GADPLUS_T1X_H5=/scratch/memoozd/GAD/data/transition1x.h5
export GADPLUS_GXTB_EXE=/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb
export GADPLUS_GXTB_PARALLEL=32 GADPLUS_ENV="$PROJECT_DIR/.venv"
export GADPLUS_GXTB_WORK_ROOT=/scratch/memoozd/gadplus/tmp/gxtb_intrinsic_topo_${SLURM_JOB_ID}
source "$PROJECT_DIR/scripts/lib/gxtb_runtime.sh"
export PYTHON PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"

run_wave() {
  local offset=$1 count=$2
  srun --ntasks="$count" --cpus-per-task=32 --cpu-bind=cores \
    bash -c 'task=$((SLURM_PROCID + '"$offset"')); "$PYTHON" '"$PROJECT_DIR"'/scripts/t1x_gxtb_sella_endpoint_score.py --sella-root /dev/null --competitive-root "$SOURCE_ROOT" --native-label-root '"$LABEL_ROOT"' --output-root '"$OUT_ROOT"' --task-id "$task" --gxtb-executable "$GADPLUS_GXTB_EXE" --parallel "$GADPLUS_GXTB_PARALLEL" --fmax 0.03'
}

for offset in 0 48 96 144 192; do run_wave "$offset" 48; done
run_wave 240 47
echo "run_root=$OUT_ROOT"
