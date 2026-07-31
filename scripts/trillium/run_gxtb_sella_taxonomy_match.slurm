#!/bin/bash
# Matched Sella baseline for the GAD taxonomy cohort: same 24 labels, noise,
# seed, and g-xTB executable.  Full direct Hessian supplied at every step.
# Twelve independent 16-core ranks occupy one 192-core Trillium CPU node.
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=gxtb_sella_taxonomy
#SBATCH --output=/scratch/memoozd/gadplus/logs/gxtb_sella_taxonomy_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/gxtb_sella_taxonomy_%j.err

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

OUT_ROOT="$GADPLUS_SCRATCH/runs/gxtb_sella_taxonomy_${SLURM_JOB_ID}"
mkdir -p "$OUT_ROOT"
export OUT_ROOT PYTHON

run_one() {
  local sample="$1"
  "$PYTHON" -u scripts/sella_smoke.py \
    --backend gxtb --method gxTB --sample-indices "$sample" \
    --start-geometry noised_ts --noise 0.25 --seed 42 \
    --max-steps 100 --fmax 0.01 --apply-eckart \
    --delta0 0.1 --gamma 0.4 --diag-every 1 --n-workers 1 \
    --output-dir "$OUT_ROOT/sella_s${sample}"
}

run_task() {
  local samples=(41 66 84 96 108 110 112 131 143 150 186 206 216 217 225 227 251 258 263 264 266 274 278 286)
  local a=$((SLURM_PROCID)) b=$((SLURM_PROCID + 12))
  run_one "${samples[$a]}"
  run_one "${samples[$b]}"
}
export -f run_one run_task

srun --ntasks=12 --ntasks-per-node=12 --cpus-per-task=16 --cpu-bind=cores \
  bash -c 'run_task'
