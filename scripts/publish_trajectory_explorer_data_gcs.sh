#!/usr/bin/env bash
# Publish the immutable data layer used by the Git-deployed Cloud Run explorer.
set -euo pipefail

BUCKET_URI="${1:?usage: $0 gs://BUCKET}"
DATA="/scratch/memoozd/gadplus"
GCLOUD="${GCLOUD:-$DATA/tools/google-cloud-sdk/bin/gcloud}"

env -u PYTHONPATH "$GCLOUD" storage cp --verbosity=error \
    "$DATA/analysis/trajectory-explorer/index.duckdb" "$BUCKET_URI/index.duckdb"

for root in \
    "$DATA/runs/t1x-gxtb-grid-regular_gad-0p10-1984934" \
    "$DATA/runs/t1x-gxtb-grid-competitive-0p10-1984935" \
    "$DATA/runs/t1x-gxtb-grid-competitive_subspace-0p10-1984936" \
    "$DATA/runs/t1x-gxtb-grid-sella-0p10-1984937" \
    "$DATA/runs/t1x-gxtb-grid-regular_gad-0p20-1984938" \
    "$DATA/runs/t1x-gxtb-grid-competitive-0p20-1984939" \
    "$DATA/runs/t1x-gxtb-grid-competitive_subspace-0p20-1984940" \
    "$DATA/runs/t1x-gxtb-grid-sella-0p20-1984941" \
    "$DATA/runs/t1x-gxtb-grid-regular_gad-1p00-1984943" \
    "$DATA/runs/t1x-gxtb-grid-competitive-1p00-1984944" \
    "$DATA/runs/t1x-gxtb-grid-competitive_subspace-1p00-1984945" \
    "$DATA/runs/t1x-gxtb-grid-sella-1p00-1984946"
    "$DATA/runs/t1x-gxtb-grid-regular_gad-0p50-1990765"
    "$DATA/runs/t1x-gxtb-grid-competitive-0p50-1990767"
    "$DATA/runs/t1x-gxtb-grid-competitive_subspace-0p50-1990769"
    "$DATA/runs/t1x-gxtb-grid-sella-0p50-1990771"
    "$DATA/runs/t1x-gxtb-grid-regular_gad-2p00-1990773"
    "$DATA/runs/t1x-gxtb-grid-competitive-2p00-1990775"
    "$DATA/runs/t1x-gxtb-grid-competitive_subspace-2p00-1990777"
    "$DATA/runs/t1x-gxtb-grid-sella-2p00-1990779"
do
    env -u PYTHONPATH "$GCLOUD" storage rsync --recursive --verbosity=error \
        "$root" "$BUCKET_URI/runs/${root##*/}"
done
