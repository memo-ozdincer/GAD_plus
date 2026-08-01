#!/bin/bash
#SBATCH --job-name=lj7_sella_progression
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/lj7_sella_progression_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/lj7_sella_progression_%j.err

# Matched analytic-LJ7 Sella comparison: 48 paired starts in each of the
# saddle/pushed panels at 0.10, 0.20, and 0.40 sigma.  Each task has one
# analytic 21x21 Hessian workload and runs independently; no nested pool.
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5

PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
OUT_ROOT=/scratch/memoozd/gadplus/runs/lj7-sella-progression-${SLURM_JOB_ID}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PROJECT_DIR/src"
PYTHON="$PROJECT_DIR/.venv/bin/python"
mkdir -p "$OUT_ROOT"

"$PYTHON" "$PROJECT_DIR/scripts/lj_method_progression_sweep.py" prepare --output-root "$OUT_ROOT"
run_wave() {
  local offset=$1 count=$2
  srun --ntasks="$count" --cpus-per-task=1 --cpu-bind=cores bash -c \
    '"$PYTHON" "$PROJECT_DIR/scripts/lj_method_progression_sweep.py" worker --output-root "$OUT_ROOT" --task-id "$((SLURM_PROCID + '"$offset"'))" --methods sella --noises 0.10,0.20,0.40 --n-samples 48 --euler-max-steps 8000'
}
run_wave 0 192
run_wave 192 96
"$PYTHON" "$PROJECT_DIR/scripts/lj_method_progression_sweep.py" aggregate --output-root "$OUT_ROOT" --expected-tasks 288
echo "output_root=$OUT_ROOT"
