#!/bin/bash
#SBATCH --job-name=t1x_grid_raw
#SBATCH --account=def-aspuru
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/t1x_grid_raw_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/t1x_grid_raw_%j.err

# One cell of the fully paired Transition1x/g-xTB comparison grid.  Submit
# with FAMILY={regular_gad,competitive,competitive_subspace,sella}, NOISE,
# MAX_STEPS, and RUN_LABEL. Every family uses seed 20260727+1001*sample_id,
# hence receives exactly the same Cartesian Gaussian displacement at a given
# noise. The node layout is six independent 32-core g-xTB Hessian jobs/node.
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5

: "${FAMILY:?regular_gad, competitive, competitive_subspace, or sella}"
: "${NOISE:?Cartesian component noise in Angstrom}"
: "${MAX_STEPS:?update budget}"
: "${RUN_LABEL:?stable output label}"
case "$FAMILY" in regular_gad|competitive|competitive_subspace|sella) ;; *) echo "unknown FAMILY=$FAMILY" >&2; exit 2;; esac

PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
export SAMPLE_IDS=$(seq -s, 0 286)
OUT_ROOT=/scratch/memoozd/gadplus/runs/${RUN_LABEL}-${SLURM_JOB_ID}
export PROJECT_DIR OUT_ROOT FAMILY NOISE MAX_STEPS RUN_LABEL
export GADPLUS_SCRATCH=/scratch/memoozd/gadplus
export GADPLUS_T1X_H5=/scratch/memoozd/GAD/data/transition1x.h5
export GADPLUS_GXTB_EXE=/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb
export GADPLUS_GXTB_PARALLEL=32 GADPLUS_ENV="$PROJECT_DIR/.venv"
export GADPLUS_GXTB_WORK_ROOT=/scratch/memoozd/gadplus/tmp/gxtb_grid_${RUN_LABEL}_${SLURM_JOB_ID}
source "$PROJECT_DIR/scripts/lib/gxtb_runtime.sh"
export PYTHON PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"

run_task() {
  local task=$1 base_seed=$((20260727 + task))
  case "$FAMILY" in
    competitive|competitive_subspace)
      "$PYTHON" "$PROJECT_DIR/scripts/t1x_intrinsic_gxtb_pilot.py" worker \
        --output-root "$OUT_ROOT" --task-id "$task" --sample-ids "$SAMPLE_IDS" \
        --noise-levels "$NOISE" --skip-endpoints \
        --gxtb-executable "$GADPLUS_GXTB_EXE" --parallel "$GADPLUS_GXTB_PARALLEL" \
        --max-steps "$MAX_STEPS" --step-fraction 0.01 --spectral-temperature 0.01 \
        --gate-variant "$FAMILY" --record-trajectories \
        --campaign "t1x-gxtb-matched-grid-${RUN_LABEL}" --selection-stage evaluation
      ;;
    regular_gad)
      "$PYTHON" -u "$PROJECT_DIR/scripts/gad_smoke.py" \
        --backend gxtb --method gxTB --sample-indices "$task" \
        --start-geometry noised_ts --noise "$NOISE" --seed "$base_seed" \
        --n-steps "$MAX_STEPS" --dt 0.003 --force-threshold 0.03 --force-criterion fmax \
        --max-atom-disp 0.35 --n-workers 1 \
        --output-dir "$OUT_ROOT/task_$(printf '%03d' "$task")"
      ;;
    sella)
      "$PYTHON" -u "$PROJECT_DIR/scripts/sella_smoke.py" \
        --backend gxtb --method gxTB --sample-indices "$task" \
        --start-geometry noised_ts --noise "$NOISE" --seed "$base_seed" \
        --max-steps "$MAX_STEPS" --fmax 0.03 --apply-eckart --delta0 0.1 --gamma 0.4 \
        --diag-every 1 --n-workers 1 \
        --output-dir "$OUT_ROOT/task_$(printf '%03d' "$task")"
      ;;
  esac
}
export -f run_task
run_wave() {
  local offset=$1 count=$2
  srun --ntasks="$count" --cpus-per-task=32 --cpu-bind=cores \
    bash -c 'run_task $((SLURM_PROCID + '"$offset"'))'
}
for offset in 0 48 96 144 192; do run_wave "$offset" 48; done
run_wave 240 47
if [[ "$FAMILY" == competitive || "$FAMILY" == competitive_subspace ]]; then
  "$PYTHON" "$PROJECT_DIR/scripts/t1x_intrinsic_gxtb_pilot.py" aggregate \
    --output-root "$OUT_ROOT" --expected-tasks 287 --sample-ids "$SAMPLE_IDS" \
    --noise-levels "$NOISE"
fi
echo "family=$FAMILY noise=$NOISE max_steps=$MAX_STEPS run_root=$OUT_ROOT"
