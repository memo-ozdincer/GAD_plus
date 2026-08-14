#!/bin/bash
# Wait on trig-login01 for a keeper, then launch the supervised HIP resume.
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "usage: $0 JOB_ID RUN_ROOT" >&2
  exit 2
fi
job_id=$1
run_root=$2
state_file=/scratch/memoozd/gadplus/allocations/hip_h100_${job_id}.json
campaign_log=/scratch/memoozd/gadplus/logs/hip_cs2_reserved_${job_id}_v2.out
supervisor_log=/scratch/memoozd/gadplus/logs/hip_cs2_supervisor_${job_id}_v2.out

while true; do
  state=$(/opt/slurm/bin/squeue -h -j "$job_id" -o '%T')
  case "$state" in
    RUNNING) break ;;
    PENDING) sleep 30 ;;
    '') echo "keeper job $job_id disappeared before launch" >&2; exit 1 ;;
    *) echo "keeper job $job_id entered unexpected state $state" >&2; exit 1 ;;
  esac
done

for _ in $(seq 1 60); do
  [[ -s "$state_file" ]] && break
  sleep 2
done
[[ -s "$state_file" ]] || { echo "keeper state missing: $state_file" >&2; exit 1; }
node=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hostname"])' "$state_file")

ssh -o BatchMode=yes -o ConnectTimeout=20 "$node" \
  "cd /scratch/memoozd/GAD/GAD_plus && nohup env KEEPER_JOB_ID='$job_id' RUN_ROOT='$run_root' CAMPAIGN_LOG='$campaign_log' bash scripts/supervise_reserved_hip_campaign.sh </dev/null >'$supervisor_log' 2>&1 &"
echo "launched keeper=$job_id node=$node run_root=$run_root"
