#!/bin/bash
# Create the lightweight post-run exporter environment below the scratch root.
set -euo pipefail

PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
OBS_ENV=/scratch/memoozd/GAD/.venv-wandb

uv venv "$OBS_ENV"
uv pip install --python "$OBS_ENV/bin/python" \
  -r "$PROJECT_DIR/requirements-observability.txt"
echo "observability_python=$OBS_ENV/bin/python"
