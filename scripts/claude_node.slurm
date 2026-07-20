#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/claude_node_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/claude_node_%j.err
#
# CPU-only node reservation for running Claude Code on a compute node.
# No GPU — Claude Code needs no VRAM.
#
# WORKFLOW
# --------
# 1. Submit:   sbatch scripts/claude_node.slurm
# 2. Get node: squeue --me        (wait for RUNNING, note the NODELIST column)
# 3. Unlock:   unlock-compute --here <nodename>   (run ON the login node)
# 4. SSH in:   ssh <nodename>      (direct SSH; allowed while you hold the node)
# 5. Run:      source ~/.cluster_env && claude    (proxy auto-injected on :8888)
#
# Bump --time above (up to 7-00:00:00) and resubmit for a longer hold;
# the sleep below already covers 3 days.

echo "========================================================"
echo " Claude node: $(hostname)"
echo " Job ID:      $SLURM_JOB_ID"
echo " Start:       $(date)"
echo "========================================================"
echo
echo "On the LOGIN node, run:   unlock-compute --here $(hostname)"
echo "Then:                     ssh $(hostname)"
echo "On the node:              source ~/.cluster_env && claude"
echo
echo "Holding node..."

sleep 259200  # 3 days (job still ends at --time, whichever is first)
