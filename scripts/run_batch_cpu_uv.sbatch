#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J scine_cpu
#SBATCH -p batch_singlenode
#SBATCH --cpus-per-task=30
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/GAD_plus/outslurm/slurm-%j.out

set -euo pipefail

cd /lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/GAD_plus

export GADPLUS_PROJECT_DIR="$PWD"
export GADPLUS_SCRATCH_DIR="$PWD"
export GADPLUS_VENV="$PWD/.venv"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ "$#" -eq 0 ]]; then
  echo "Usage: sbatch $0 <uv-run-args...>" >&2
  exit 2
fi

exec uv run "$@"
