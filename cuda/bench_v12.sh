#!/usr/bin/env bash
# bench_v12.sh -- Build + run the v12 packed-trit kernel benchmark.
#
# USAGE (run on Ryzen box, e.g. via SSH):
#   ssh xasko@<ryzen-host> 'cd /path/to/ter/cuda && ./bench_v12.sh > results.csv'
#
# Or, while logged in:
#   cd ~/ter/cuda && ./bench_v12.sh | tee results.csv
#
# PREREQS:
#   - NVCC in PATH (CUDA 12.x recommended).
#   - An Ampere or newer GPU (sm_80+). Default arch is sm_86 (RTX 3090).
#   - msys64 bash on the Windows host or a regular Linux bash.
#
# OUTPUT FORMAT (stdout):
#   kernel,shape,ms_median,gb_per_s,bw_eff_percent
#   v11,K=2048_N=2048,...
#   v12,K=2048_N=2048,...
#   ... (one row per kernel/shape pair, covering the 5 Llama 1B GEMV shapes)
#
# Build diagnostics go to stderr (incl. device name + reported peak HBM BW).
#
# Override knobs (env vars):
#   ARCH=sm_86   target SM arch (sm_80 for A100; sm_89 for Ada; sm_90 for Hopper)
#   ITERS=200    timed iterations per (kernel, shape)
#   WARMUP=20    untimed warmup iterations
#   NVCC=nvcc    path to nvcc binary

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${SCRIPT_DIR}/ter_cuda_forward_packed_v12.cu"
BIN="${SCRIPT_DIR}/ter_cuda_forward_packed_v12"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_86}"
ITERS="${ITERS:-200}"
WARMUP="${WARMUP:-20}"

if [[ ! -f "$SRC" ]]; then
    echo "ERROR: source not found: $SRC" >&2
    exit 1
fi

# Build only if source is newer than binary (cheap rebuild guard).
if [[ ! -x "$BIN" ]] || [[ "$SRC" -nt "$BIN" ]]; then
    echo "[bench_v12] Compiling for $ARCH ..." >&2
    "$NVCC" -O3 -std=c++17 -arch="$ARCH" \
        -Xcompiler -Wall -Xptxas -O3 \
        "$SRC" -o "$BIN"
fi

echo "[bench_v12] Running $BIN iters=$ITERS warmup=$WARMUP" >&2
"$BIN" "$ITERS" "$WARMUP"
