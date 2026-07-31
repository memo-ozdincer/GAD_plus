#!/bin/bash
# Packed small-batch comparison: 3 candidates x 3 GAD timesteps, plus Sella.
# Each independent trajectory has 32 xTB workers; six are packed per
# Trillium CPU node to use all 192 cores without nested Python workers.
#SBATCH --account=def-aspuru
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --job-name=gxtb_small_compare
#SBATCH --output=/scratch/memoozd/gadplus/logs/gxtb_small_compare_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/gxtb_small_compare_%j.err

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

OUT_ROOT="$GADPLUS_SCRATCH/runs/gxtb_small_compare_${SLURM_JOB_ID}"
mkdir -p "$OUT_ROOT"
export OUT_ROOT PYTHON

run_task() {
  local kind sample dt
  case "$SLURM_PROCID" in
    0) kind=gad;   sample=2;  dt=0.0005 ;;
    1) kind=gad;   sample=17; dt=0.0005 ;;
    2) kind=gad;   sample=43; dt=0.0005 ;;
    3) kind=gad;   sample=2;  dt=0.003  ;;
    4) kind=gad;   sample=17; dt=0.003  ;;
    5) kind=gad;   sample=43; dt=0.003  ;;
    6) kind=gad;   sample=2;  dt=0.01   ;;
    7) kind=gad;   sample=17; dt=0.01   ;;
    8) kind=gad;   sample=43; dt=0.01   ;;
    9) kind=sella; sample=2             ;;
   10) kind=sella; sample=17            ;;
   11) kind=sella; sample=43            ;;
    *) echo "Unexpected Slurm rank: $SLURM_PROCID" >&2; exit 2 ;;
  esac
  if [[ "$kind" == gad ]]; then
    exec "$PYTHON" -u scripts/gad_smoke.py \
      --backend gxtb --method gxTB --sample-indices "$sample" \
      --start-geometry noised_ts --noise 0.10 --seed 42 \
      --n-steps 100 --dt "$dt" --force-threshold 0.01 --force-criterion fmax \
      --n-workers 1 --output-dir "$OUT_ROOT/gad_s${sample}_dt${dt}"
  fi
  exec "$PYTHON" -u scripts/sella_smoke.py \
    --backend gxtb --method gxTB --sample-indices "$sample" \
    --start-geometry noised_ts --noise 0.10 --seed 42 \
    --max-steps 20 --fmax 0.01 --apply-eckart --delta0 0.1 --gamma 0.4 \
    --diag-every 1 --n-workers 1 --output-dir "$OUT_ROOT/sella_s${sample}"
}
export -f run_task

srun --ntasks=12 --ntasks-per-node=6 --cpus-per-task=32 --cpu-bind=cores \
  bash -c 'run_task'
