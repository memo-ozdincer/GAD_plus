#!/bin/bash
#SBATCH --job-name=t1x_grid_topology
#SBATCH --account=def-aspuru
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/t1x_grid_topology_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/t1x_grid_topology_%j.err

# Native two-branch endpoint score for one completed matched raw campaign.
# Submit with FAMILY={regular_gad,competitive,competitive_subspace,sella} and
# SOURCE_ROOT=/scratch/.../raw-root.  The selector applies the identical local
# candidate criterion (projected index one and fmax < 0.03 eV A^-1) for every
# family; it is deliberately separate from the raw optimizer.  One 32-core
# g-xTB calculation is used by each independent task, six per 192-core node.
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5

: "${FAMILY:?regular_gad, competitive, competitive_subspace, or sella}"
: "${SOURCE_ROOT:?completed matched raw campaign root}"
case "$FAMILY" in regular_gad|competitive|competitive_subspace|sella) ;; *) echo "unknown FAMILY=$FAMILY" >&2; exit 2;; esac

PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
LABEL_ROOT=${LABEL_ROOT:-/scratch/memoozd/gadplus/runs/t1x-gxtb-native-labels-287-1961128/native_labels}
OUT_ROOT=/scratch/memoozd/gadplus/runs/t1x-gxtb-grid-topology-${FAMILY}-$(basename "$SOURCE_ROOT")-${SLURM_JOB_ID}
export PROJECT_DIR LABEL_ROOT OUT_ROOT FAMILY SOURCE_ROOT
export GADPLUS_SCRATCH=/scratch/memoozd/gadplus
export GADPLUS_T1X_H5=/scratch/memoozd/GAD/data/transition1x.h5
export GADPLUS_GXTB_EXE=/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb
export GADPLUS_GXTB_PARALLEL=32 GADPLUS_ENV="$PROJECT_DIR/.venv"
export GADPLUS_GXTB_WORK_ROOT=/scratch/memoozd/gadplus/tmp/gxtb_grid_topology_${FAMILY}_${SLURM_JOB_ID}
source "$PROJECT_DIR/scripts/lib/gxtb_runtime.sh"
export PYTHON PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"

run_wave() {
  local offset=$1 count=$2
  srun --ntasks="$count" --cpus-per-task=32 --cpu-bind=cores bash -c '
    task=$((SLURM_PROCID + '"$offset"'))
    if [[ "$FAMILY" == competitive || "$FAMILY" == competitive_subspace ]]; then
      input=(--sella-root /dev/null --competitive-root "$SOURCE_ROOT")
    else
      input=(--sella-root "$SOURCE_ROOT")
    fi
    "$PYTHON" "$PROJECT_DIR/scripts/t1x_gxtb_sella_endpoint_score.py" \
      "${input[@]}" --native-label-root "$LABEL_ROOT" --output-root "$OUT_ROOT" \
      --task-id "$task" --gxtb-executable "$GADPLUS_GXTB_EXE" \
      --parallel "$GADPLUS_GXTB_PARALLEL" --fmax 0.03
  '
}
for offset in 0 48 96 144 192; do run_wave "$offset" 48; done
run_wave 240 47
echo "family=$FAMILY source_root=$SOURCE_ROOT run_root=$OUT_ROOT"
