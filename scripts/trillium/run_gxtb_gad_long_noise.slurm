#!/bin/bash
# Long GAD robustness check: 3 candidates x 2 noise levels, packed as six
# independent 32-core trajectories on one 192-core Trillium CPU node.
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --job-name=gxtb_gad_long_noise
#SBATCH --output=/scratch/memoozd/gadplus/logs/gxtb_gad_long_noise_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/gxtb_gad_long_noise_%j.err

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

OUT_ROOT="$GADPLUS_SCRATCH/runs/gxtb_gad_long_noise_${SLURM_JOB_ID}"
mkdir -p "$OUT_ROOT"
export OUT_ROOT PYTHON

run_task() {
  local sample noise
  case "$SLURM_PROCID" in
    0) sample=2;  noise=0.25 ;;
    1) sample=17; noise=0.25 ;;
    2) sample=43; noise=0.25 ;;
    3) sample=2;  noise=0.50 ;;
    4) sample=17; noise=0.50 ;;
    5) sample=43; noise=0.50 ;;
    *) echo "Unexpected Slurm rank: $SLURM_PROCID" >&2; exit 2 ;;
  esac
  local adaptive_args=()
  if [[ "${GAD_ADAPTIVE_DT:-0}" == 1 ]]; then
    adaptive_args=(--use-adaptive-dt)
  fi
  exec "$PYTHON" -u scripts/gad_smoke.py \
    --backend gxtb --method gxTB --sample-indices "$sample" \
    --start-geometry noised_ts --noise "$noise" --seed 42 \
    --n-steps 1000 --dt 0.003 --force-threshold 0.01 --force-criterion fmax \
    --descent-until-nneg "${DESCENT_UNTIL_NNEG:-0}" \
    --max-atom-disp "${GAD_MAX_ATOM_DISP:-0.35}" "${adaptive_args[@]}" \
    --n-workers 1 --output-dir "$OUT_ROOT/gad_s${sample}_noise${noise}"
}
export -f run_task

srun --ntasks=6 --ntasks-per-node=6 --cpus-per-task=32 --cpu-bind=cores \
  bash -c 'run_task'
