#!/bin/bash
# The research digest owns the MiMo server on DGX Sparks 3+4 (added 2026-11-13).
#
#   prepare   ExecStartPre of ai-digest-mimo-tunnel.service. When the server on spark4 is not
#             running, restart the RPC worker on spark3 before the server starts. After a spark4
#             reboot the worker still holds the dead connection from before the reboot; it serves
#             one client at a time, so a new server would wait for it forever (seen 2026-11-11).
#   ensure    Run by the digest before each cycle (LLM_START_COMMAND). Starts the digest's
#             units, waits until the recording proxy lists the model, and repairs a stalled
#             server once. Exit 0 = ready.
#   stop [--force]  Stop the server cleanly: the laptop units first (so they cannot restart it),
#             then the server on spark4 and the worker on spark3 (systemd units and any tmux
#             sessions named mimo/rpcsrv). Refuses with exit 3 while the digest's daily run is
#             using MiMo, unless --force. An unreachable Spark counts as already stopped.
#             Used by mimo-spark when its window closes and by shs/sds before powering off.
#
# Units (laptop, ~/.config/systemd/user): ai-digest-mimo-worker-link (ssh spark3, keeps the
# worker), ai-digest-mimo-tunnel (127.0.0.1:18021 -> spark4:8084), ai-digest-mimo-proxy
# (127.0.0.1:18121, ledger label mimo-spark34-mxfp4). Remote units: ai-digest-mimo-rpc
# (spark3), ai-digest-mimo-server (spark4, ctx 32768).
# Other tools use this server through 'ensure' and stop it only through 'stop' (2026-11-17:
# the server should close when the mimo-spark window closes).
set -uo pipefail

HEAD=spark4; WORKER=spark3; HPORT=8084
PROXY_URL=http://127.0.0.1:18121/v1/models
MODEL=MiMo-V2.6-Flash-MOPD-MXFP4
UNITS=(ai-digest-mimo-worker-link.service ai-digest-mimo-tunnel.service ai-digest-mimo-proxy.service)
WAIT_SECONDS=${MIMO_START_WAIT_SECONDS:-1200}   # per attempt; a cold load takes ~5 min
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)

AGENT_LOG=$(dirname "$(readlink -f "$0")")/agent.log

say() { echo "[mimo-spark-service] $(date '+%F %T') $*" >&2; }

ready() { curl -sf -m 10 "$PROXY_URL" 2>/dev/null | grep -q "\"$MODEL\""; }

prepare() {
    if "${SSH[@]}" "$HEAD" 'systemctl --user is-active -q ai-digest-mimo-server.service'; then
        exit 0
    fi
    # Port 8084 held by anything else (for example a hand-started tmux server) blocks ours.
    local holder
    holder=$("${SSH[@]}" "$HEAD" "ss -ltnpH 'sport = :$HPORT'") || exit 1
    if [[ -n "$holder" ]]; then
        say "port $HPORT on $HEAD is held by another process, not ai-digest-mimo-server: $holder"
        exit 1
    fi
    say "server on $HEAD is not running: restarting the RPC worker on $WORKER first"
    "${SSH[@]}" "$WORKER" 'systemctl --user restart ai-digest-mimo-rpc.service' || exit 1
    sleep 3
}

start_units() {
    systemctl --user reset-failed "${UNITS[@]}" 2>/dev/null
    systemctl --user start "${UNITS[@]}"
}

wait_ready() {
    local waited=0
    while (( waited < WAIT_SECONDS )); do
        ready && return 0
        sleep 15; waited=$((waited + 15))
    done
    return 1
}

ensure() {
    if ready; then say "ready: $MODEL at $PROXY_URL"; exit 0; fi
    say "not ready; starting ${UNITS[*]}"
    start_units || say "systemctl start reported an error (the units retry on their own)"
    if wait_ready; then say "ready: $MODEL at $PROXY_URL"; exit 0; fi
    # One repair: stop the server and tunnel; prepare then gives it a fresh worker.
    say "not ready after ${WAIT_SECONDS}s; restarting the server on $HEAD and the worker on $WORKER"
    systemctl --user stop ai-digest-mimo-tunnel.service
    "${SSH[@]}" "$HEAD" 'systemctl --user stop ai-digest-mimo-server.service'
    start_units
    if wait_ready; then say "ready after repair: $MODEL at $PROXY_URL"; exit 0; fi
    say "still not ready; check: ssh $HEAD 'journalctl --user -u ai-digest-mimo-server -n 30 --no-pager'"
    exit 1
}

# True while the digest daemon is between "Starting primary model" and the end of that cycle.
digest_busy() {
    systemctl --user is-active -q ai-research-agent.service || return 1
    local last
    last=$(tac "$AGENT_LOG" 2>/dev/null | grep -m1 -E \
        'Starting primary model|=== Cycle (start|complete)|Cycle failed|Agent daemon (started|stopped)')
    [[ "$last" == *"Starting primary model"* || "$last" == *"=== Cycle start"* ]]
}

# Stop a unit and a tmux session on one node and wait until the port is free.
# ssh exit 255 (node unreachable, e.g. already powered off) counts as stopped.
stop_node() {
    local node=$1 unit=$2 session=$3 port=$4 rc
    "${SSH[@]}" "$node" "systemctl --user stop $unit 2>/dev/null; tmux kill-session -t =$session 2>/dev/null
        for i in \$(seq 1 30); do [ -z \"\$(ss -ltnH 'sport = :$port')\" ] && exit 0; sleep 1; done; exit 1"
    rc=$?
    case $rc in
        0) say "stopped on $node (port $port free)" ;;
        255) say "$node unreachable; treating it as already stopped" ;;
        *) say "$node: port $port still in use after 30 s"; return 1 ;;
    esac
}

stop() {
    if [[ "${1:-}" != --force ]] && digest_busy; then
        say "the research digest's daily run is using MiMo right now; leaving the server running"
        exit 3
    fi
    systemctl --user stop "${UNITS[@]}"
    local failed=0
    stop_node "$HEAD" ai-digest-mimo-server.service mimo "$HPORT" || failed=1
    stop_node "$WORKER" ai-digest-mimo-rpc.service rpcsrv 50052 || failed=1
    exit $failed
}

case "${1:-}" in
    prepare) prepare ;;
    ensure) ensure ;;
    stop) stop "${2:-}" ;;
    *) echo "usage: $0 prepare|ensure|stop [--force]" >&2; exit 2 ;;
esac
