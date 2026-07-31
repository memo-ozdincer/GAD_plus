#!/bin/bash
#SBATCH --job-name=t1x_sella_topology
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/t1x_sella_topology_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/t1x_sella_topology_%j.err
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5
PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
SELLA_ROOT=/scratch/memoozd/gadplus/runs/t1x-gxtb-sella-heldout-1957786
LABEL_ROOT=/scratch/memoozd/gadplus/runs/t1x-intrinsic-gxtb-heldout-gates-1953766/native_labels
OUT_ROOT=/scratch/memoozd/gadplus/runs/t1x-gxtb-sella-topology-${SLURM_JOB_ID}
export GADPLUS_SCRATCH=/scratch/memoozd/gadplus
export GADPLUS_T1X_H5=/scratch/memoozd/GAD/data/transition1x.h5
export GADPLUS_GXTB_EXE=/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb
export GADPLUS_GXTB_PARALLEL=16 GADPLUS_ENV="$PROJECT_DIR/.venv"
export GADPLUS_GXTB_WORK_ROOT=/scratch/memoozd/gadplus/tmp/gxtb_sella_topology_${SLURM_JOB_ID}
source "$PROJECT_DIR/scripts/lib/gxtb_runtime.sh"
export PYTHON PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"
for offset in 0 12; do
  srun --ntasks=12 --ntasks-per-node=12 --cpus-per-task=16 --cpu-bind=cores \
    bash -c 'task=$((SLURM_PROCID + '"$offset"')); "$PYTHON" '"$PROJECT_DIR"'/scripts/t1x_gxtb_sella_endpoint_score.py --sella-root '"$SELLA_ROOT"' --native-label-root '"$LABEL_ROOT"' --output-root '"$OUT_ROOT"' --task-id "$task" --gxtb-executable "$GADPLUS_GXTB_EXE" --parallel "$GADPLUS_GXTB_PARALLEL" --fmax 0.03'
done
echo "run_root=$OUT_ROOT"
