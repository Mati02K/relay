#!/usr/bin/env bash
# Build llama.cpp from source with CUDA enabled and point Relay at it.
#
# Usage:
#   ./build-llama-cuda.sh
#
# What it does:
#   1. Checks required tools are installed (git, cmake, nvcc, python3).
#   2. Clones (or fast-forwards) llama.cpp into ~/.relay/build/llama.cpp.
#   3. Configures and builds the llama-server target with -DGGML_CUDA=on.
#   4. Writes the resulting binary path into ~/.relay/config.json under
#      engine.server_bin so the next `relay start` picks it up.
#
# Idempotent: safe to re-run. Honors RELAY_HOME if set.
#
# This script intentionally does NOT install CUDA Toolkit, GCC, or cmake. If
# any of those are missing it tells you and exits.

set -euo pipefail

RELAY_HOME="${RELAY_HOME:-$HOME/.relay}"
CONFIG_PATH="$RELAY_HOME/config.json"
LLAMA_REPO="https://github.com/ggml-org/llama.cpp"
BUILD_ROOT="$RELAY_HOME/build/llama.cpp"
BIN_PATH="$BUILD_ROOT/build/bin/llama-server"
LOG_PATH="$RELAY_HOME/logs/build-llama-cuda.log"

color_red()  { printf '\033[31m%s\033[0m' "$1"; }
color_green(){ printf '\033[32m%s\033[0m' "$1"; }
color_dim()  { printf '\033[2m%s\033[0m' "$1"; }

info()  { printf '%s %s\n' "$(color_green '==>')" "$1"; }
warn()  { printf '%s %s\n' "$(color_red 'WARN:')" "$1" >&2; }
die()   { printf '%s %s\n' "$(color_red 'ERROR:')" "$1" >&2; exit 1; }

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "$1 is required but not installed. $2"
}

step_check_prereqs() {
    info "Checking prerequisites"
    need_cmd git    "Install with your package manager."
    need_cmd cmake  "Install with: sudo apt install cmake  (or your distro equivalent)"
    need_cmd python3 "python3 is required to update config.json."
    need_cmd nvcc   "Install CUDA Toolkit with: sudo apt install nvidia-cuda-toolkit  (or download from https://developer.nvidia.com/cuda-downloads)"
    need_cmd nvidia-smi "nvidia-smi not found — CUDA driver is not installed. This script is for machines with an NVIDIA GPU."

    if [ ! -f "$CONFIG_PATH" ]; then
        die "Relay config not found at $CONFIG_PATH. Run 'relay init' first."
    fi

    local nproc_count
    nproc_count="$(nproc 2>/dev/null || echo 4)"
    info "Will build with -j $nproc_count"
    PARALLEL="$nproc_count"

    info "nvcc: $(nvcc --version | head -4 | tail -1)"
    info "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
}

step_prepare_dirs() {
    info "Preparing $BUILD_ROOT"
    mkdir -p "$(dirname "$BUILD_ROOT")"
    mkdir -p "$(dirname "$LOG_PATH")"
    : > "$LOG_PATH"
}

step_clone_or_update() {
    if [ -d "$BUILD_ROOT/.git" ]; then
        info "Updating existing llama.cpp checkout"
        git -C "$BUILD_ROOT" fetch --tags --depth=1 origin HEAD 2>&1 | tee -a "$LOG_PATH"
        git -C "$BUILD_ROOT" reset --hard FETCH_HEAD 2>&1 | tee -a "$LOG_PATH"
    else
        info "Cloning llama.cpp into $BUILD_ROOT"
        git clone --depth 1 "$LLAMA_REPO" "$BUILD_ROOT" 2>&1 | tee -a "$LOG_PATH"
    fi
}

step_configure() {
    info "Configuring (cmake)"
    # GGML_CUDA: turn on the CUDA backend.
    # LLAMA_CURL: off so we don't need libcurl headers — Relay passes the
    #             model path directly, llama-server never downloads.
    # CMAKE_BUILD_TYPE=Release: full optimisations.
    cmake -S "$BUILD_ROOT" -B "$BUILD_ROOT/build" \
        -DGGML_CUDA=on \
        -DLLAMA_CURL=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        2>&1 | tee -a "$LOG_PATH"
}

step_build() {
    info "Building llama-server (this is the slow part — 5-15 minutes)"
    info "Full log: $LOG_PATH"
    cmake --build "$BUILD_ROOT/build" \
        --target llama-server \
        --config Release \
        -j "$PARALLEL" \
        2>&1 | tee -a "$LOG_PATH"
}

step_verify() {
    info "Verifying binary at $BIN_PATH"
    if [ ! -x "$BIN_PATH" ]; then
        die "Build finished but $BIN_PATH does not exist. Check $LOG_PATH."
    fi
    local devices
    devices="$("$BIN_PATH" --list-devices 2>&1 || true)"
    printf '%s\n' "$devices"
    if printf '%s' "$devices" | grep -qi cuda; then
        info "CUDA device detected $(color_green '✓')"
    else
        warn "CUDA device not listed in --list-devices output."
        warn "The binary built, but llama-server cannot see your GPU at runtime."
        warn "Possible causes: nvidia driver/toolkit version mismatch, no GPU permissions."
    fi
}

step_update_config() {
    info "Updating $CONFIG_PATH to point engine.server_bin → $BIN_PATH"
    python3 - "$CONFIG_PATH" "$BIN_PATH" <<'PY'
import json
import shutil
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
bin_path = sys.argv[2]
backup = config_path.with_suffix(config_path.suffix + ".bak")

shutil.copy2(config_path, backup)
cfg = json.loads(config_path.read_text())
cfg.setdefault("engine", {})["server_bin"] = bin_path
config_path.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"  backup: {backup}")
print(f"  set engine.server_bin = {bin_path}")
PY
}

step_summary() {
    echo
    info "$(color_green 'Done.') Next steps:"
    echo "  1. Stop the running cluster:        relay stop"
    echo "  2. Drop the cached prebuilt llama:  rm -rf ~/.relay/bin/llama.cpp-*"
    echo "  3. Start with the CUDA build:       relay start"
    echo
    color_dim "Step 2 is optional — config.json already overrides the bin path."
    echo
}

main() {
    step_check_prereqs
    step_prepare_dirs
    step_clone_or_update
    step_configure
    step_build
    step_verify
    step_update_config
    step_summary
}

main "$@"
