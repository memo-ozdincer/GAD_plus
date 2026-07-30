#!/bin/bash
#SBATCH --job-name=t1x_gad_min_prevent
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/t1x_gad_min_prevent_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/t1x_gad_min_prevent_%j.err

# Paired, prevention-only diagnostic. These are exactly the 23 index-zero
# terminals from the frozen competitive 0.20 A test-287 run. Every task is
# regenerated from its original sample-id-derived seed and initial geometry.
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5

PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
export ALL_IDS=$(seq -s, 0 286)
export MINIMUM_IDS=0,15,17,30,68,75,76,115,146,172,178,179,181,182,188,199,209,212,237,252,267,276,283
OUT_ROOT=/scratch/memoozd/gadplus/runs/t1x-gxtb-minimum-prevention-${SLURM_JOB_ID}
export GADPLUS_SCRATCH=/scratch/memoozd/gadplus
export GADPLUS_T1X_H5=/scratch/memoozd/GAD/data/transition1x.h5
export GADPLUS_GXTB_EXE=/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb
export GADPLUS_GXTB_PARALLEL=32 GADPLUS_ENV="$PROJECT_DIR/.venv"
export GADPLUS_GXTB_WORK_ROOT=/scratch/memoozd/gadplus/tmp/gxtb_min_prevent_${SLURM_JOB_ID}
source "$PROJECT_DIR/scripts/lib/gxtb_runtime.sh"
export PYTHON PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"

run_wave() {
  local offset=$1 count=$2
  srun --ntasks="$count" --cpus-per-task=32 --cpu-bind=cores \
    bash -c 'IFS=, read -r -a ids <<< "$MINIMUM_IDS"; idx=$((SLURM_PROCID + '"$offset"')); count=${#ids[@]}; sample=${ids[$((idx % count))]}; variant=competitive; if (( idx >= count )); then variant=competitive_subspace; fi; "$PYTHON" '"$PROJECT_DIR"'/scripts/t1x_intrinsic_gxtb_pilot.py worker --output-root '"$OUT_ROOT"'/"$variant" --task-id "$sample" --sample-ids "$ALL_IDS" --noise-levels 0.20 --skip-endpoints --gxtb-executable "$GADPLUS_GXTB_EXE" --parallel "$GADPLUS_GXTB_PARALLEL" --max-steps 300 --step-fraction 0.01 --spectral-temperature 0.01 --gate-variant "$variant" --record-trajectories --campaign t1x-gxtb-minimum-prevention --selection-stage calibration'
}

for offset in 0 6 12 18 24 30 36 42; do
  count=6
  if (( offset == 42 )); then count=4; fi
  run_wave "$offset" "$count"
done

for variant in competitive competitive_subspace; do
  "$PYTHON" "$PROJECT_DIR/scripts/t1x_intrinsic_gxtb_pilot.py" aggregate \
    --output-root "$OUT_ROOT/$variant" --expected-tasks 23
done
echo "run_root=$OUT_ROOT"
