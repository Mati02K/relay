#!/usr/bin/env bash
# Scapegoat network jitter for the `jitter` scenario — run on the Mac (MLX) worker.
#
# The coordinator measures jitter as the RTT of its /health probe to this worker,
# smoothed by an EMA. Adding outbound delay here inflates that RTT, so j_w rises
# and (with the jitter signal on) the scheduler routes away from this node.
#
# macOS uses dummynet (dnctl + pfctl), NOT Linux `tc`. Requires sudo.
#
# Usage
# -----
#   sudo ./mac_jitter.sh start [DELAY_MS]   # default 200ms
#   sudo ./mac_jitter.sh stop
#
# After `start`, wait ~12s for the coordinator's jitter EMA to settle, then run
# the jitter scenario. ALWAYS run `stop` afterwards to restore networking.
#
# Note: pf/dummynet syntax varies a little across macOS versions; if `start`
# errors, check `man pfctl`/`man dnctl` for your version. Verify it took effect by
# watching j_w climb on the coordinator's /v1/workers for this node.
set -euo pipefail

CMD="${1:-}"
DELAY_MS="${2:-200}"
ANCHOR="relay_jitter"
PIPE=1

require_root() {
    if [[ "${EUID}" -ne 0 ]]; then
        echo "Run with sudo: sudo $0 ${CMD:-start} ..." >&2
        exit 1
    fi
}

case "${CMD}" in
    start)
        require_root
        # 1) a dummynet pipe that delays packets
        dnctl pipe "${PIPE}" config delay "${DELAY_MS}ms"
        # 2) a pf anchor that pushes this host's outbound traffic into the pipe
        echo "dummynet out all pipe ${PIPE}" | pfctl -a "${ANCHOR}" -f -
        # 3) hook the anchor into the running ruleset and enable pf
        (pfctl -sr 2>/dev/null || true; \
         echo "dummynet-anchor \"${ANCHOR}\""; \
         echo "anchor \"${ANCHOR}\"") | pfctl -f - 2>/dev/null || true
        pfctl -e 2>/dev/null || true
        echo "Jitter ON: ~${DELAY_MS}ms outbound delay (anchor '${ANCHOR}', pipe ${PIPE})."
        echo "Wait ~12s for the coordinator EMA, then run the jitter scenario."
        ;;
    stop)
        require_root
        pfctl -a "${ANCHOR}" -F all 2>/dev/null || true
        dnctl -q flush 2>/dev/null || true
        pfctl -f /etc/pf.conf 2>/dev/null || true
        pfctl -d 2>/dev/null || true
        echo "Jitter OFF: anchor flushed, pipe cleared, pf reset to /etc/pf.conf."
        ;;
    *)
        echo "usage: sudo $0 start [delay_ms] | sudo $0 stop" >&2
        exit 1
        ;;
esac
