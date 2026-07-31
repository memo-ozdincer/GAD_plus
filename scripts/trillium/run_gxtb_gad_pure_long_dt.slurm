#!/bin/bash
# Standard projected GAD only: long fixed-dt test at 0.25 A noise.
# Six independent g-xTB trajectories pack one 192-core Trillium CPU node.
# No descent phase, adaptive timestep, preconditioning, or Sella refinement.
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --job-name=gxtb_gad_pure_long_dt
#SBATCH --output=/scratch/memoozd/gadplus/logs/gxtb_gad_pure_long_dt_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/gxtb_gad_pure_long_dt_%j.err

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

OUT_ROOT="$GADPLUS_SCRATCH/runs/gxtb_gad_pure_long_dt_${SLURM_JOB_ID}"
mkdir -p "$OUT_ROOT"
export OUT_ROOT PYTHON

run_task() {
  local sample dt
  case "$SLURM_PROCID" in
    0) sample=2;  dt=0.0005 ;;
    1) sample=17; dt=0.0005 ;;
    2) sample=43; dt=0.0005 ;;
    3) sample=2;  dt=0.0010 ;;
    4) sample=17; dt=0.0010 ;;
    5) sample=43; dt=0.0010 ;;
    *) echo "Unexpected Slurm rank: $SLURM_PROCID" >&2; exit 2 ;;
  esac
  exec "$PYTHON" -u scripts/gad_smoke.py \
    --backend gxtb --method gxTB --sample-indices "$sample" \
    --start-geometry noised_ts --noise 0.25 --seed 42 \
    --n-steps 5000 --dt "$dt" --force-threshold 0.01 --force-criterion fmax \
    --descent-until-nneg 0 --max-atom-disp 0.35 \
    --n-workers 1 --output-dir "$OUT_ROOT/gad_s${sample}_dt${dt}"
}
export -f run_task

srun --ntasks=6 --ntasks-per-node=6 --cpus-per-task=32 --cpu-bind=cores \
  bash -c 'run_task'
