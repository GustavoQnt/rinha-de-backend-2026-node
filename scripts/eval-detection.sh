#!/usr/bin/env bash
# Lightweight detection eval. Runs sweep-blocks-nprobe with current
# fast/full nprobe pairs and reports the bucket / approved-mismatch counts
# that drive the FP/FN seen on the prévia.
#
# Usage:
#   scripts/eval-detection.sh                       # default sweep
#   scripts/eval-detection.sh 16 48,64,96           # fast_nprobe + full sweep
#   scripts/eval-detection.sh 24 64,96,128          # explore wider
#
# Output columns of interest:
#   approved_mm  -> # of queries whose fraud/legit verdict changes vs ground
#                   truth. This is what produces FPs/FNs on the leaderboard.
#   bucket_mm    -> total bucket disagreements (includes harmless 1<->2 etc.)
#   ms_p99       -> single-thread cold latency (no warmup / io_uring).
#                   Real prévia is typically 40-60% lower.

set -euo pipefail
cd "$(dirname "$0")/.."

FAST_NPROBE="${1:-16}"
NPROBES="${2:-48,64,96}"

BIN=native/rinha-server/target/release/sweep-blocks-nprobe.exe
if [ ! -x "$BIN" ]; then
  echo "building sweep-blocks-nprobe..."
  (cd native/rinha-server && cargo build --release --bin sweep-blocks-nprobe >/dev/null 2>&1)
fi

echo "fast_nprobe=${FAST_NPROBE}  full_nprobe sweep=${NPROBES}"
echo

"$BIN" \
  --refs resources/references.bin \
  --blocks resources/references.ivf-blocks-K4096.bin \
  --requests test/test-data.requests.ndjson \
  --expected-buckets test/test-data.expected-buckets.ndjson \
  --nprobes "$NPROBES" \
  --fast-nprobe "$FAST_NPROBE" 2>&1 | tail -n +14
