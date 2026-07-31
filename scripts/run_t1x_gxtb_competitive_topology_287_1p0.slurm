#!/bin/bash
#SBATCH --job-name=t1x_comp_topo_1p0
#SBATCH --account=def-aspuru
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/t1x_comp_topo_1p0_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/t1x_comp_topo_1p0_%j.err

# Endpoint-topology phase for the 105 local TS candidates from the matched
# 1.00 A, 2,000-step competitive-GAD arm.  Labels are noise-independent and
# are reused from the already completed calculator-native labelling pass.
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5
PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
SOURCE_ROOT=/scratch/memoozd/gadplus/runs/t1x-gxtb-competitive-287-raw-1p0-1961751
LABEL_ROOT=/scratch/memoozd/gadplus/runs/t1x-gxtb-native-labels-287-1961128/native_labels
OUT_ROOT=/scratch/memoozd/gadplus/runs/t1x-gxtb-competitive-287-topology-1p0-${SLURM_JOB_ID}
export GADPLUS_SCRATCH=/scratch/memoozd/gadplus
export GADPLUS_T1X_H5=/scratch/memoozd/GAD/data/transition1x.h5
export GADPLUS_GXTB_EXE=/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb
export GADPLUS_GXTB_PARALLEL=32 GADPLUS_ENV="$PROJECT_DIR/.venv"
export GADPLUS_GXTB_WORK_ROOT=/scratch/memoozd/gadplus/tmp/gxtb_comp_topo_1p0_${SLURM_JOB_ID}
source "$PROJECT_DIR/scripts/lib/gxtb_runtime.sh"
export PYTHON PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"
run_wave() {
  local offset=$1 count=$2
  srun --ntasks="$count" --cpus-per-task=32 --cpu-bind=cores \
    bash -c 'task=$((SLURM_PROCID + '"$offset"')); "$PYTHON" '"$PROJECT_DIR"'/scripts/t1x_gxtb_sella_endpoint_score.py --sella-root /dev/null --competitive-root '"$SOURCE_ROOT"' --native-label-root '"$LABEL_ROOT"' --output-root '"$OUT_ROOT"' --task-id "$task" --gxtb-executable "$GADPLUS_GXTB_EXE" --parallel "$GADPLUS_GXTB_PARALLEL" --fmax 0.03'
}
run_wave 0 48
run_wave 48 48
run_wave 96 9
echo "run_root=$OUT_ROOT"
