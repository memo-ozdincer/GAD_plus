#!/usr/bin/env bash
# Non-destructive source/assets transfer helper for the Narval -> Trillium
# migration.  It never uses rsync --delete.
#
# Usage:
#   ./scripts/migrate_to_trillium.sh
#   MIGRATE_EXTERNAL=1 GADPLUS_GXTB_DIR=/path/to/g-xtb ./scripts/migrate_to_trillium.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TRILLIUM_HOST="${TRILLIUM_HOST:-trillium.alliancecan.ca}"
TRILLIUM_USER="${TRILLIUM_USER:-$USER}"
TRILLIUM_ROOT="${TRILLIUM_ROOT:-/project/rrg-aspuru/memoozd}"
DEST="$TRILLIUM_USER@$TRILLIUM_HOST:$TRILLIUM_ROOT"

echo "Source:      $PROJECT_DIR"
echo "Destination: $DEST"
ssh "$TRILLIUM_USER@$TRILLIUM_HOST" "mkdir -p '$TRILLIUM_ROOT/GAD_plus'"

rsync -a \
  --exclude '.git/' --exclude '.venv/' --exclude 'g-xtb/' \
  --exclude 'runs/' --exclude 'outputs/' --exclude 'mlruns/' \
  --exclude 'figures_new/' --exclude 'narval_runs/' \
  "$PROJECT_DIR/" "$DEST/GAD_plus/"

if [[ "${MIGRATE_EXTERNAL:-0}" != "1" ]]; then
    cat <<'EOF'

Repository copied.  External assets were intentionally skipped.
To transfer them, rerun with MIGRATE_EXTERNAL=1 and set the source paths:
  GADPLUS_HIP_DIR=/path/to/hip
  GADPLUS_T1X_DIR=/path/to/transition1x
  GADPLUS_T1X_H5=/path/to/transition1x.h5
  GADPLUS_GXTB_DIR=/path/to/g-xtb
EOF
    exit 0
fi

copy_dir() {
    local source=$1 destination=$2 label=$3
    if [[ -z "$source" || ! -d "$source" ]]; then
        echo "Skipping $label: source directory is not set or missing" >&2
        return
    fi
    ssh "$TRILLIUM_USER@$TRILLIUM_HOST" "mkdir -p '$destination'"
    rsync -a "$source/" "$TRILLIUM_USER@$TRILLIUM_HOST:$destination/"
}

copy_dir "${GADPLUS_HIP_DIR:-}" "$TRILLIUM_ROOT/hip" "HIP checkout"
copy_dir "${GADPLUS_T1X_DIR:-}" "$TRILLIUM_ROOT/transition1x" "Transition1x checkout"
copy_dir "${GADPLUS_GXTB_DIR:-}" "$TRILLIUM_ROOT/third_party/g-xtb" "g-xTB installation"

if [[ -n "${GADPLUS_T1X_H5:-}" && -f "$GADPLUS_T1X_H5" ]]; then
    ssh "$TRILLIUM_USER@$TRILLIUM_HOST" "mkdir -p '$TRILLIUM_ROOT/data'"
    rsync -a "$GADPLUS_T1X_H5" "$TRILLIUM_USER@$TRILLIUM_HOST:$TRILLIUM_ROOT/data/transition1x.h5"
else
    echo "Skipping Transition1x HDF5: set GADPLUS_T1X_H5 to an existing file" >&2
fi

echo "Migration transfer complete.  Follow docs/TRILLIUM_MIGRATION.md on Trillium."
