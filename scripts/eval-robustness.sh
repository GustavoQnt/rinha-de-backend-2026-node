#!/usr/bin/env bash
# Robustness eval: resamples the existing local fixture into multiple synthetic
# distributions and runs sweep-blocks-nprobe against each. The ground-truth
# bucket labels stay valid because refs don't change — we only re-stratify the
# query mix to simulate dataset drift (different fraud_rate, edge_rate, seeds).
#
# Usage:
#   scripts/eval-robustness.sh             # default sweep
#   scripts/eval-robustness.sh 16 64       # fast=16 full=64
#   FAST=24 FULL=96 scripts/eval-robustness.sh

set -euo pipefail
cd "$(dirname "$0")/.."

FAST="${1:-${FAST:-16}}"
FULL="${2:-${FULL:-64}}"

# (label, fraud_rate, edge_rate, seed)
DATASETS=(
  "baseline    0.47  0.012  1"
  "low_fraud   0.30  0.012  2"
  "high_fraud  0.60  0.012  3"
  "high_edge   0.47  0.030  4"
  "low_edge    0.47  0.005  5"
  "alt_seed    0.44  0.012  9999"
)

OUT=test/robustness
mkdir -p "$OUT"

BIN=native/rinha-server/target/release/sweep-blocks-nprobe.exe
if [ ! -x "$BIN" ]; then
  echo "building sweep-blocks-nprobe..." >&2
  (cd native/rinha-server && cargo build --release --bin sweep-blocks-nprobe >/dev/null 2>&1)
fi

echo "== Robustness eval (fast_nprobe=$FAST full_nprobe=$FULL) =="
printf '%-12s | %-6s | %-6s | %-11s | %-9s | %s\n' "dataset" "fraud" "edge" "approved_mm" "bucket_mm" "p99(ms)"
echo "---------------------------------------------------------------------------"

for spec in "${DATASETS[@]}"; do
  read -r label fraud_rate edge_rate seed <<< "$spec"
  prefix="$OUT/$label"
  reqs="$prefix.requests.ndjson"
  buckets="$prefix.expected-buckets.ndjson"

  if [ ! -f "$reqs" ] || [ ! -f "$buckets" ]; then
    node scripts/resample-test-data.js \
      --out-prefix "$prefix" \
      --fraud-rate "$fraud_rate" \
      --edge-rate "$edge_rate" \
      --seed "$seed" \
      --total 54100 2>/dev/null
  fi

  csv=$("$BIN" \
    --refs resources/references.bin \
    --blocks resources/references.ivf-blocks-K4096.bin \
    --requests "$reqs" \
    --expected-buckets "$buckets" \
    --nprobes "$FULL" \
    --fast-nprobe "$FAST" 2>/dev/null | tail -1)
  approved=$(echo "$csv" | awk -F, '{print $5}')
  buckmm=$(echo "$csv" | awk -F, '{print $4}')
  p99=$(echo "$csv" | awk -F, '{print $11}')

  printf '%-12s | %-6s | %-6s | %-11s | %-9s | %s\n' "$label" "$fraud_rate" "$edge_rate" "$approved" "$buckmm" "$p99"
done
