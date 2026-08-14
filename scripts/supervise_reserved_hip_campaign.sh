#!/bin/bash
# Keep an SSH-launched HIP campaign within the lifetime of its Slurm keeper.
set -euo pipefail

WORK_ROOT=/scratch/memoozd/GAD/GAD_plus
KEEPER_JOB_ID=${KEEPER_JOB_ID:?set KEEPER_JOB_ID to the active reservation}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to the immutable campaign root}
CAMPAIGN_LOG=${CAMPAIGN_LOG:-/scratch/memoozd/gadplus/logs/hip_cs2_reserved_${KEEPER_JOB_ID}.out}
CAMPAIGN_SCRIPT=${CAMPAIGN_SCRIPT:-scripts/resume_hip_missing_on_reserved_h100.sh}

cd "$WORK_ROOT"
if [[ $(/opt/slurm/bin/squeue -h -j "$KEEPER_JOB_ID" -o '%T') != RUNNING ]]; then
  echo "keeper job $KEEPER_JOB_ID is not RUNNING" >&2
  exit 2
fi

# SSH commands on Trillium compute nodes start non-login shells and therefore
# do not define Lmod's `module` function. Run the frozen scientific launcher
# through a login shell; it then performs its own module purge/load sequence.
setsid env KEEPER_JOB_ID="$KEEPER_JOB_ID" RUN_ROOT="$RUN_ROOT" \
  bash -l "$CAMPAIGN_SCRIPT" </dev/null \
  >>"$CAMPAIGN_LOG" 2>&1 &
campaign_pid=$!

terminate_campaign() {
  kill -TERM -- "-$campaign_pid" 2>/dev/null || true
}
trap terminate_campaign INT TERM EXIT

while kill -0 "$campaign_pid" 2>/dev/null; do
  if [[ $(/opt/slurm/bin/squeue -h -j "$KEEPER_JOB_ID" -o '%T') != RUNNING ]]; then
    echo "keeper job $KEEPER_JOB_ID ended; terminating campaign $campaign_pid" \
      >>"$CAMPAIGN_LOG"
    terminate_campaign
    break
  fi
  sleep 30
done
wait "$campaign_pid"
