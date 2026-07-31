#!/usr/bin/env bash
# Shared runtime for the maintained g-xTB Slurm templates.
# Source this after setting PROJECT_DIR and, optionally, GADPLUS_SCRATCH.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this file from a Slurm script; do not execute it directly." >&2
    exit 2
fi

: "${PROJECT_DIR:?PROJECT_DIR must be set before sourcing gxtb_runtime.sh}"
: "${GADPLUS_ENV:=$PROJECT_DIR/.venv}"
: "${GADPLUS_SCRATCH:=${SCRATCH:-/tmp/$USER}/gadplus}"
: "${GADPLUS_T1X_H5:=$PROJECT_DIR/data/transition1x.h5}"
if [[ -z "${GADPLUS_GXTB_EXE:-}" ]]; then
    # Prefer a clean external installation; retain the earlier in-tree clone
    # as a compatibility fallback until the migration is complete.
    if [[ -x "$PROJECT_DIR/../third_party/g-xtb/xtb-6.7.1/bin/xtb" ]]; then
        GADPLUS_GXTB_EXE="$PROJECT_DIR/../third_party/g-xtb/xtb-6.7.1/bin/xtb"
    else
        GADPLUS_GXTB_EXE="$PROJECT_DIR/g-xtb/xtb-6.7.1/bin/xtb"
    fi
fi
: "${GADPLUS_GXTB_PARALLEL:=${SLURM_CPUS_PER_TASK:-1}}"

if [[ ! -x "$GADPLUS_ENV/bin/python" ]]; then
    echo "Missing Python environment: $GADPLUS_ENV" >&2
    exit 2
fi
if [[ ! -x "$GADPLUS_GXTB_EXE" ]]; then
    echo "Missing g-xTB executable: $GADPLUS_GXTB_EXE" >&2
    exit 2
fi
if [[ ! -f "$GADPLUS_T1X_H5" ]]; then
    echo "Missing Transition1x HDF5: $GADPLUS_T1X_H5" >&2
    exit 2
fi

source "$GADPLUS_ENV/bin/activate"
PYTHON="$GADPLUS_ENV/bin/python"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true
# g-xTB owns the allocation through --parallel; avoid nested BLAS threads.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="${SLURM_TMPDIR:-$GADPLUS_SCRATCH/tmp}"
export GADPLUS_GXTB_WORK_ROOT="${GADPLUS_GXTB_WORK_ROOT:-$GADPLUS_SCRATCH/debug/gxtb}"
mkdir -p "$GADPLUS_SCRATCH/runs" "$GADPLUS_SCRATCH/logs" "$GADPLUS_GXTB_WORK_ROOT" "$TMPDIR"
