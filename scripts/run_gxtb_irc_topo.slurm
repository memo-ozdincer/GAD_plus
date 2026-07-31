#!/bin/bash
# Validate g-xTB candidates with the same TS and IRC_TOPO gates.
# Pass one or more GAD/Sella summary parquet paths as arguments.
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/gxtb_irc_topo_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/gxtb_irc_topo_%j.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6

PROJECT_DIR="${PROJECT_DIR:-/lustre06/project/6033559/memoozd/GAD_plus}"
export GADPLUS_SCRATCH="${GADPLUS_SCRATCH:-/lustre07/scratch/memoozd/gadplus}"
cd "$PROJECT_DIR"
source scripts/lib/gxtb_runtime.sh
OUT="$GADPLUS_SCRATCH/runs/gxtb_irc_topo_${SLURM_JOB_ID}"

if (( $# == 0 )); then
  echo "Pass at least one summary parquet from run_gxtb_gad or run_gxtb_sella." >&2
  exit 2
fi
"$PYTHON" -u scripts/gxtb_irc_topo.py \
  --summary "$@" --output-dir "$OUT" \
  --gxtb-executable "$GADPLUS_GXTB_EXE"
