#!/bin/bash
#SBATCH --job-name=t1x_gad_topo
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/t1x_gad_topo_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/t1x_gad_topo_%j.err
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5
PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
SOURCE_ROOT=/scratch/memoozd/gadplus/runs/t1x-gxtb-regular-gad-287-raw-1977950
LABEL_ROOT=/scratch/memoozd/gadplus/runs/t1x-gxtb-native-labels-287-1961128/native_labels
OUT_ROOT=/scratch/memoozd/gadplus/runs/t1x-gxtb-regular-gad-287-topology-${SLURM_JOB_ID}
export GADPLUS_SCRATCH=/scratch/memoozd/gadplus
export GADPLUS_T1X_H5=/scratch/memoozd/GAD/data/transition1x.h5
export GADPLUS_GXTB_EXE=/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb
export GADPLUS_GXTB_PARALLEL=32 GADPLUS_ENV="$PROJECT_DIR/.venv"
export GADPLUS_GXTB_WORK_ROOT=/scratch/memoozd/gadplus/tmp/gxtb_regular_gad_topo_${SLURM_JOB_ID}
source "$PROJECT_DIR/scripts/lib/gxtb_runtime.sh"
export PYTHON PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"
srun --ntasks=2 --cpus-per-task=32 --cpu-bind=cores \
  bash -c '"$PYTHON" '"$PROJECT_DIR"'/scripts/t1x_gxtb_sella_endpoint_score.py --sella-root '"$SOURCE_ROOT"' --native-label-root '"$LABEL_ROOT"' --output-root '"$OUT_ROOT"' --task-id "$SLURM_PROCID" --gxtb-executable "$GADPLUS_GXTB_EXE" --parallel "$GADPLUS_GXTB_PARALLEL" --fmax 0.03'
echo "run_root=$OUT_ROOT"
