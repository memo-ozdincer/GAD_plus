#!/bin/bash
# GAD-only recovery test from the 12 minimum-like taxonomy endpoints.
# Each endpoint is nudged in both signs along its softest projected mode, then
# resumed with standard projected GAD.  Twelve 16-core ranks fill one CPU node.
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=gxtb_gad_minrescue
#SBATCH --output=/scratch/memoozd/gadplus/logs/gxtb_gad_minrescue_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/gxtb_gad_minrescue_%j.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5

PROJECT_DIR="/scratch/memoozd/GAD/GAD_plus"
export GADPLUS_SCRATCH="/scratch/memoozd/gadplus"
export GADPLUS_T1X_H5="/scratch/memoozd/GAD/data/transition1x.h5"
export GADPLUS_GXTB_EXE="/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb"
export GADPLUS_GXTB_PARALLEL=16
cd "$PROJECT_DIR"
source scripts/lib/gxtb_runtime.sh

SOURCE_ROOT="/scratch/memoozd/gadplus/runs/gxtb_gad_taxonomy_1946008"
OUT_ROOT="$GADPLUS_SCRATCH/runs/gxtb_gad_minrescue_${SLURM_JOB_ID}"
mkdir -p "$OUT_ROOT"
export SOURCE_ROOT OUT_ROOT PYTHON

run_one() {
  local sample="$1" sign="$2"
  "$PYTHON" -u scripts/gad_minimum_rescue.py \
    --source-dir "$SOURCE_ROOT/gad_s${sample}" --sample-index "$sample" \
    --sign "$sign" --amplitude "${NUDGE_AMPLITUDE:-0.05}" --n-steps 500 --dt 0.003 \
    --force-threshold 0.01 --output-dir "$OUT_ROOT/s${sample}_sign${sign}"
}

run_task() {
  local samples=(84 96 112 206 225 227 251 263 264 266 274 286)
  local sample="${samples[$SLURM_PROCID]}"
  run_one "$sample" -1
  run_one "$sample" 1
}
export -f run_one run_task

srun --ntasks=12 --ntasks-per-node=12 --cpus-per-task=16 --cpu-bind=cores \
  bash -c 'run_task'
