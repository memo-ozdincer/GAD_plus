#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J gly_pt_orca
#SBATCH -p cpu
#SBATCH --array=0-79%10
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -t 1-00:00:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/GAD_plus/outslurm/gly_pt_orca-%A_%a.out

set -euo pipefail

REPO=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/GAD_plus
ORCA_DIR=/lustre/fsw/portfolios/nvr/users/anburger/software/orca_6_1_1
MPI_DIR=/cm/shared/apps/openmpi4/gcc/4.1.5
INPUT_LIST="$REPO/runs/glycine_pt_scan/orca_input_list.txt"
OUTPUT_DIR="$REPO/runs/glycine_pt_scan/orca_outputs"
WORK_ROOT="$REPO/runs/glycine_pt_scan/orca_work"

export PATH="$ORCA_DIR:$MPI_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ORCA_DIR/lib:$MPI_DIR/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$REPO"

INPUT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$INPUT_LIST")
if [[ -z "${INPUT:-}" ]]; then
  echo "No input for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

STEM=$(basename "$INPUT" .inp)
WORK_DIR="$WORK_ROOT/$STEM"
mkdir -p "$OUTPUT_DIR" "$WORK_DIR"
rm -rf "$WORK_DIR"/*
cp "$INPUT" "$WORK_DIR/$STEM.inp"

cd "$WORK_DIR"
echo "[$(date)] Starting ORCA for $STEM on $(hostname)"
echo "ORCA: $ORCA_DIR/orca"
echo "MPI: $MPI_DIR"
if ldd "$ORCA_DIR/orca_scfgrad_mpi" | grep -q 'not found'; then
  ldd "$ORCA_DIR/orca_scfgrad_mpi" >&2
  exit 1
fi
"$ORCA_DIR/orca" "$STEM.inp" > "$STEM.out"
echo "[$(date)] Finished ORCA for $STEM"

shopt -s nullglob
cp "$STEM".* "$OUTPUT_DIR"/
