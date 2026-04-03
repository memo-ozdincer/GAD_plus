#!/bin/bash
# Setup GADplus environment on Trillium (SciNet HPC)
#
# Usage:
#   bash scripts/setup_env.sh
#
# Prerequisites:
#   - Module system available (StdEnv/2023, python/3.11.5, cuda/12.6)
#   - uv installed (pip install uv)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "=== GADplus Environment Setup ==="
echo "Project: $PROJECT_DIR"
echo "Venv:    $VENV_DIR"

# Load modules
module purge
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.6

# Create venv with uv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    uv venv "$VENV_DIR" --python python3.11
fi

source "$VENV_DIR/bin/activate"

# Install GADplus in editable mode
echo "Installing GADplus..."
uv pip install -e "$PROJECT_DIR"

# Install local dependencies (HIP, transition1x)
HIP_DIR="/project/rrg-aspuru/memoozd/hip"
T1X_DIR="/project/rrg-aspuru/memoozd/transition1x"

if [ -d "$HIP_DIR" ]; then
    echo "Installing HIP from $HIP_DIR..."
    uv pip install -e "$HIP_DIR"
else
    echo "WARNING: HIP not found at $HIP_DIR"
fi

if [ -d "$T1X_DIR" ]; then
    echo "Installing transition1x from $T1X_DIR..."
    uv pip install -e "$T1X_DIR"
else
    echo "WARNING: transition1x not found at $T1X_DIR"
fi

# Install optional analysis deps
echo "Installing analysis dependencies..."
uv pip install -e "$PROJECT_DIR[analysis]"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
