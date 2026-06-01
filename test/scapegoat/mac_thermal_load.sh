#!/usr/bin/env bash
# Scapegoat thermal load for the `thermal` scenario — run on the Mac (MLX) worker.
#
# Apple Silicon shares one thermal envelope across CPU and GPU, so hammering all
# CPU cores heats it. The worker's telemetry/thermal/apple.py reads
# NSProcessInfo.thermalState; once it leaves "nominal" it reports theta_w > 0
# ("fair" -> 0.35, "serious" -> 0.75), and with the thermal signal on the
# scheduler routes away from this node.
#
# Run it a few minutes BEFORE the thermal scenario and keep it running — Macs
# cool aggressively, so it takes sustained load to leave nominal. No sudo needed.
# Ctrl-C to stop (it cools back down on its own).
#
# Watch the state while it heats (separate terminal):
#   while true; do pmset -g therm | grep -i 'CPU_Speed_Limit'; sleep 2; done
set -euo pipefail

N="$(sysctl -n hw.ncpu)"
echo "Heating thermal envelope with ${N} CPU stressors…"
echo "Keep running for a few minutes; theta_w rises once the Mac leaves 'nominal'."
echo "Tip: also let the thermal test's inference traffic hit this node (GPU heat). Ctrl-C to stop."

pids=()
cleanup() {
    kill "${pids[@]}" 2>/dev/null || true
    echo
    echo "Stopped — cooling down."
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 "${N}"); do
    # busy loop per core; redirect to /dev/null so it just burns CPU
    yes > /dev/null &
    pids+=("$!")
done

wait
