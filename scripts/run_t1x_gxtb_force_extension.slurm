#!/bin/bash
#SBATCH --job-name=t1x_gad_force_10k
#SBATCH --account=def-aspuru
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/t1x_gad_force_10k_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/t1x_gad_force_10k_%j.err

# The 22 index-one/high-force terminals from the 300-step 0.20 A campaign,
# plus two 0.03-A near-threshold index-one controls (19,79), fill four
# 192-core nodes exactly. The vector field is unchanged; only the permitted
# evaluation budget and force reporting threshold are extended.
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5

PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
export ALL_IDS=$(seq -s, 0 286)
export EXTENSION_IDS=59,64,80,85,116,120,123,127,131,160,166,197,208,213,218,221,234,246,273,279,281,282,19,79
OUT_ROOT=/scratch/memoozd/gadplus/runs/t1x-gxtb-force-extension-${SLURM_JOB_ID}
export GADPLUS_SCRATCH=/scratch/memoozd/gadplus
export GADPLUS_T1X_H5=/scratch/memoozd/GAD/data/transition1x.h5
export GADPLUS_GXTB_EXE=/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb
export GADPLUS_GXTB_PARALLEL=32 GADPLUS_ENV="$PROJECT_DIR/.venv"
export GADPLUS_GXTB_WORK_ROOT=/scratch/memoozd/gadplus/tmp/gxtb_force_10k_${SLURM_JOB_ID}
source "$PROJECT_DIR/scripts/lib/gxtb_runtime.sh"
export PYTHON PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"

srun --ntasks=24 --ntasks-per-node=6 --cpus-per-task=32 --cpu-bind=cores \
  bash -c 'IFS=, read -r -a ids <<< "$EXTENSION_IDS"; sample=${ids[$SLURM_PROCID]}; "$PYTHON" '"$PROJECT_DIR"'/scripts/t1x_intrinsic_gxtb_pilot.py worker --output-root '"$OUT_ROOT"' --task-id "$sample" --sample-ids "$ALL_IDS" --noise-levels 0.20 --skip-endpoints --gxtb-executable "$GADPLUS_GXTB_EXE" --parallel "$GADPLUS_GXTB_PARALLEL" --max-steps 10000 --search-fmax 0.01 --strict-fmax 0.01 --step-fraction 0.01 --spectral-temperature 0.01 --gate-variant competitive --record-trajectories --campaign t1x-gxtb-force-extension --selection-stage calibration'

"$PYTHON" "$PROJECT_DIR/scripts/t1x_intrinsic_gxtb_pilot.py" aggregate \
  --output-root "$OUT_ROOT" --expected-tasks 24
echo "run_root=$OUT_ROOT"
