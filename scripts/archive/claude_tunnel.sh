#!/bin/bash
# claude_tunnel.sh — give a Narval compute node outbound internet for Claude Code.
#
# Why this exists: `unlock-compute --here <node>` starts the HTTP proxy fine,
# but its reverse-tunnel step fails on Narval. It hardcodes
# `-i ~/.ssh/id_ed25519` (no such key here) and `-F /dev/null`, and the latter
# disables host-based auth — the very method that lets a login node SSH to a
# compute node on Narval. This script starts the proxy (if needed) and the
# reverse tunnel using host-based auth, which works.
#
# Run ON the login node:   bash scripts/claude_tunnel.sh <node>
# Then:                    ssh <node>  ->  source ~/.cluster_env && claude
#
# Tear down:               pkill -f 'ssh -N.*-R 8888'   (and optionally the proxy)

set -uo pipefail
NODE="${1:?usage: claude_tunnel.sh <compute-node>}"
PORT="${PROXY_PORT:-8888}"
PROXY="$HOME/.local/bin/http-connect-proxy"

# 1. Proxy on the login node (idempotent)
if ! ss -tlnp 2>/dev/null | grep -q "127.0.0.1:${PORT} "; then
  nohup python3 "$PROXY" "$PORT" </dev/null >"/tmp/hcp-${PORT}.log" 2>&1 &
  disown
  sleep 1
fi
ss -tlnp 2>/dev/null | grep -q "127.0.0.1:${PORT} " \
  && echo "proxy: listening on 127.0.0.1:${PORT}" \
  || { echo "proxy: FAILED to start (see /tmp/hcp-${PORT}.log)"; exit 1; }

# 2. Reverse tunnel compute:PORT -> login:PORT via host-based auth.
#    NOTE: no -F /dev/null, no -i — those break host-based auth on Narval.
pkill -f "ssh -N.*-R ${PORT}.*${NODE}" 2>/dev/null
sleep 1
setsid ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=no \
  -o ControlMaster=no \
  -R "${PORT}:127.0.0.1:${PORT}" "$NODE" \
  </dev/null >"/tmp/tunnel-${NODE}.log" 2>&1 &
disown
sleep 2
if pgrep -f "ssh -N.*-R ${PORT}.*${NODE}" >/dev/null; then
  echo "tunnel: ${NODE}:${PORT} -> login:${PORT} up"
  echo "test:   ssh ${NODE} 'HTTPS_PROXY=http://127.0.0.1:${PORT} curl -sI https://github.com | head -1'"
else
  echo "tunnel: FAILED (see /tmp/tunnel-${NODE}.log)"; cat "/tmp/tunnel-${NODE}.log"; exit 1
fi
