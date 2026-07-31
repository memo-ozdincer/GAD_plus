#!/bin/bash
# Long standard-GAD continuations from the index-1 branches recovered by the
# +0.15 A soft-mode kick.  Six 32-core trajectories fill one CPU node.
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --job-name=gxtb_gad_kick_long
#SBATCH --output=/scratch/memoozd/gadplus/logs/gxtb_gad_kick_long_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/gxtb_gad_kick_long_%j.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5

PROJECT_DIR="/scratch/memoozd/GAD/GAD_plus"
export GADPLUS_SCRATCH="/scratch/memoozd/gadplus"
export GADPLUS_T1X_H5="/scratch/memoozd/GAD/data/transition1x.h5"
export GADPLUS_GXTB_EXE="/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb"
export GADPLUS_GXTB_PARALLEL=32
cd "$PROJECT_DIR"
source scripts/lib/gxtb_runtime.sh

SOURCE_ROOT="/scratch/memoozd/gadplus/runs/gxtb_gad_taxonomy_1946008"
OUT_ROOT="$GADPLUS_SCRATCH/runs/gxtb_gad_kick_long_${SLURM_JOB_ID}"
mkdir -p "$OUT_ROOT"
export SOURCE_ROOT OUT_ROOT PYTHON

run_one() {
  local sample="$1" dt="$2" tag="$3"
  "$PYTHON" -u scripts/gad_minimum_rescue.py \
    --source-dir "$SOURCE_ROOT/gad_s${sample}" --sample-index "$sample" \
    --sign 1 --amplitude 0.15 --n-steps 5000 --dt "$dt" \
    --force-threshold 0.01 --output-dir "$OUT_ROOT/${tag}"
}

run_task() {
  case "$SLURM_PROCID" in
    0) run_one 84  0.003 s84_dt003 ;;
    1) run_one 84  0.001 s84_dt001 ;;
    2) run_one 112 0.003 s112_dt003 ;;
    3) run_one 206 0.003 s206_dt003 ;;
    4) run_one 227 0.003 s227_dt003 ;;
    5) run_one 274 0.003 s274_dt003 ;;
    *) exit 2 ;;
  esac
}
export -f run_one run_task

srun --ntasks=6 --ntasks-per-node=6 --cpus-per-task=32 --cpu-bind=cores \
  bash -c 'run_task'
