#!/bin/bash
# Trillium CPU template: full-Hessian IRC_TOPO validation of gate-passing TSs.
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/gxtb_irc_topo_%A_%a.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/gxtb_irc_topo_%A_%a.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5

if (( $# == 0 )); then
    echo "Pass one or more GAD/Sella summary parquet paths." >&2
    exit 2
fi
PROJECT_DIR="/scratch/memoozd/GAD/GAD_plus"
export GADPLUS_SCRATCH="/scratch/memoozd/gadplus"
export GADPLUS_T1X_H5="/scratch/memoozd/GAD/data/transition1x.h5"
export GADPLUS_GXTB_EXE="/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb"
export GADPLUS_GXTB_PARALLEL=96
cd "$PROJECT_DIR"
source scripts/lib/gxtb_runtime.sh

OUT="$GADPLUS_SCRATCH/runs/gxtb_irc_topo_${SLURM_JOB_ID}"
candidate_args=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  candidate_args=(--candidate-index "$SLURM_ARRAY_TASK_ID")
fi
"$PYTHON" -u scripts/gxtb_irc_topo.py --summary "$@" --output-dir "$OUT" \
  --fmax "${TS_FMAX:-0.01}" "${candidate_args[@]}"
