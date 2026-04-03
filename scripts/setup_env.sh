#!/bin/bash
# Setup GAD_plus environment on Alliance Canada HPC clusters
#
# Usage (Narval — primary):
#   bash scripts/setup_env.sh
#
# Usage (Trillium — secondary):
#   CLUSTER=trillium bash scripts/setup_env.sh
#
# After setup, activate with:
#   source .venv/bin/activate

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
CLUSTER="${CLUSTER:-narval}"

echo "=== GAD_plus Environment Setup ==="
echo "Cluster: $CLUSTER"
echo "Project: $PROJECT_DIR"
echo "Venv:    $VENV_DIR"

# ---- Cluster-specific paths ----
if [ "$CLUSTER" = "narval" ]; then
    PARENT="/lustre06/project/6033559/memoozd"
    SCRATCH="/lustre07/scratch/memoozd"
    PYTHON_MOD="python/3.11"
elif [ "$CLUSTER" = "trillium" ]; then
    PARENT="/project/rrg-aspuru/memoozd"
    SCRATCH="/scratch/memoozd"
    PYTHON_MOD="python/3.11.5"
else
    echo "ERROR: Unknown cluster '$CLUSTER'. Use 'narval' or 'trillium'."
    exit 1
fi

HIP_DIR="$PARENT/hip"
T1X_DIR="$PARENT/transition1x"

echo "HIP:     $HIP_DIR"
echo "T1x:     $T1X_DIR"
echo "Scratch: $SCRATCH"

# ---- Load modules ----
module purge
module load StdEnv/2023
module load "$PYTHON_MOD"
module load cuda/12.6

# ---- Create venv ----
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ---- Install pip + uv ----
pip install --upgrade pip
pip install uv 2>/dev/null || echo "uv already installed or unavailable, using pip"

# ---- Install GAD_plus ----
echo "Installing GAD_plus..."
if command -v uv &>/dev/null; then
    uv pip install -e "$PROJECT_DIR"
    uv pip install -e "$PROJECT_DIR[analysis]"
else
    pip install -e "$PROJECT_DIR"
    pip install -e "$PROJECT_DIR[analysis]"
fi

# ---- Install local dependencies ----
INSTALLER="pip install -e"
if command -v uv &>/dev/null; then
    INSTALLER="uv pip install -e"
fi

if [ -d "$HIP_DIR" ]; then
    echo "Installing HIP from $HIP_DIR..."
    $INSTALLER "$HIP_DIR"
else
    echo "WARNING: HIP not found at $HIP_DIR"
    echo "  Clone it: git clone <hip-repo> $HIP_DIR"
fi

if [ -d "$T1X_DIR" ]; then
    echo "Installing transition1x from $T1X_DIR..."
    $INSTALLER "$T1X_DIR"
else
    echo "WARNING: transition1x not found at $T1X_DIR"
    echo "  Clone it: git clone <t1x-repo> $T1X_DIR"
fi

# ---- Create scratch dirs ----
echo "Creating scratch directories..."
mkdir -p "$SCRATCH/gadplus/runs"
mkdir -p "$SCRATCH/gadplus/mlruns"
mkdir -p "$SCRATCH/gadplus/tmp"
mkdir -p "$SCRATCH/gadplus/logs"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo ""
echo "Verify data exists:"
echo "  ls $PARENT/models/hip_v2.ckpt"
echo "  ls $PARENT/data/transition1x.h5"
echo ""
echo "Quick test:"
echo "  python -c 'from gadplus.core.convergence import is_ts_converged; print(\"OK\")'"
