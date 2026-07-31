#!/bin/bash
# 24 deterministic test-set samples, 0.25 A noise, standard projected GAD.
# Twelve independent 16-core g-xTB ranks fill one 192-core Trillium CPU node.
# Each rank performs two serial trajectories: no nested Python workers.
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=gxtb_gad_taxonomy
#SBATCH --output=/scratch/memoozd/gadplus/logs/gxtb_gad_taxonomy_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/gxtb_gad_taxonomy_%j.err

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

OUT_ROOT="$GADPLUS_SCRATCH/runs/gxtb_gad_taxonomy_${SLURM_JOB_ID}"
mkdir -p "$OUT_ROOT"
export OUT_ROOT PYTHON

# Fixed RNG(20260727) sample from the 287-entry test split.  Excludes the
# earlier exploratory labels 2, 17, and 43 to avoid re-counting them.
run_one() {
  local sample="$1"
  "$PYTHON" -u scripts/gad_smoke.py \
    --backend gxtb --method gxTB --sample-indices "$sample" \
    --start-geometry noised_ts --noise 0.25 --seed 42 \
    --n-steps "${GAD_N_STEPS:-300}" --dt "${GAD_DT:-0.003}" --force-threshold 0.01 --force-criterion fmax \
    --descent-until-nneg 0 --max-atom-disp "${GAD_MAX_ATOM_DISP:-0.35}" --n-workers 1 \
    --output-dir "$OUT_ROOT/gad_s${sample}"
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

"$PYTHON" scripts/analyze_gad_trajectories.py \
  --glob "$OUT_ROOT/**/traj_*.parquet" \
  --output "$OUT_ROOT/gad_taxonomy.csv"
