#!/usr/bin/env bash
# Full-stack local bench in WSL: build srv via docker (works around the
# compose buildkit credential bug in WSL), bring up stack via compose, wait
# /ready, run k6 ramp-arrival-rate, parse results.json, print score.
#
# Mirrors what the prévia runner does, so DELTAS between runs predict
# deltas in the real score. Absolute numbers don't (host hardware differs,
# and WSL has 2 cores total vs the runner's 2 dedicated vCPU).
#
# Usage:
#   scripts/bench-wsl.sh                          # uses docker-compose.scm.yml
#   LABEL=foo scripts/bench-wsl.sh path/to/compose.yml
#   SKIP_BUILD=1 scripts/bench-wsl.sh             # reuse last build
#   IVF_FAST=8 IVF_FULL=32 scripts/bench-wsl.sh   # override env vars in-flight
#   IVF_BBOX_SEED=1 IVF_BBOX_VISIT_CAP=32 scripts/bench-wsl.sh
#
# Each run snapshots test/results.json -> test/bench-runs/<LABEL>.json so
# multiple configs can be compared afterwards.

set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE_FILE="${1:-docker-compose.scm.yml}"
LABEL="${LABEL:-$(basename "$COMPOSE_FILE" .yml)-$(date +%H%M%S)}"
OUT_DIR=test/bench-runs
mkdir -p "$OUT_DIR"

WSL_DISTRO="${WSL_DISTRO:-Ubuntu-24.04}"
REPO_WSL=/mnt/c/Users/breno/Documents/projects/rinha-backend/rinha-de-backend-2026-node

IMAGE_TAG="rinha-srv:bench"
LB_TAG="rinha-lb:bench"

# Env-var overrides for the app service. Empty by default — falls back to
# whatever the compose file has wired up.
ENV_OVERRIDES=""
[ -n "${IVF_FAST:-}" ]            && ENV_OVERRIDES="$ENV_OVERRIDES -e IVF_FAST_NPROBE=$IVF_FAST"
[ -n "${IVF_FULL:-}" ]            && ENV_OVERRIDES="$ENV_OVERRIDES -e IVF_FULL_NPROBE=$IVF_FULL"
[ -n "${IVF_NPROBE:-}" ]          && ENV_OVERRIDES="$ENV_OVERRIDES -e IVF_NPROBE=$IVF_NPROBE"
[ -n "${IVF_BBOX_SEED:-}" ]       && ENV_OVERRIDES="$ENV_OVERRIDES -e IVF_BBOX_SEED=$IVF_BBOX_SEED"
[ -n "${IVF_BBOX_VISIT_CAP:-}" ]  && ENV_OVERRIDES="$ENV_OVERRIDES -e IVF_BBOX_VISIT_CAP=$IVF_BBOX_VISIT_CAP"

echo "== bench label='$LABEL' compose='$COMPOSE_FILE' overrides='$ENV_OVERRIDES' =="
echo

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  echo ">> building srv image"
  wsl.exe -d "$WSL_DISTRO" -- bash -lc "cd $REPO_WSL && DOCKER_BUILDKIT=1 docker build -f Dockerfile -t $IMAGE_TAG . 2>&1 | tail -3"
  echo ">> building lb image"
  wsl.exe -d "$WSL_DISTRO" -- bash -lc "cd $REPO_WSL && DOCKER_BUILDKIT=1 docker build -f Dockerfile.lb -t $LB_TAG . 2>&1 | tail -3"
fi

# Compose override that points the services at our just-built local images
# and applies the env-var overrides for this run. Written as a yml fragment.
OVERRIDE_FILE="docker-compose.bench-override.yml"
cat > "$OVERRIDE_FILE" <<EOF
services:
  app1:
    image: $IMAGE_TAG
    build: !reset null
    environment:
$( [ -n "${IVF_FAST:-}" ]           && echo "      IVF_FAST_NPROBE: \"$IVF_FAST\"" )
$( [ -n "${IVF_FULL:-}" ]           && echo "      IVF_FULL_NPROBE: \"$IVF_FULL\"" )
$( [ -n "${IVF_NPROBE:-}" ]         && echo "      IVF_NPROBE: \"$IVF_NPROBE\"" )
$( [ -n "${IVF_BBOX_SEED:-}" ]      && echo "      IVF_BBOX_SEED: \"$IVF_BBOX_SEED\"" )
$( [ -n "${IVF_BBOX_VISIT_CAP:-}" ] && echo "      IVF_BBOX_VISIT_CAP: \"$IVF_BBOX_VISIT_CAP\"" )
  app2:
    image: $IMAGE_TAG
    build: !reset null
    environment:
$( [ -n "${IVF_FAST:-}" ]           && echo "      IVF_FAST_NPROBE: \"$IVF_FAST\"" )
$( [ -n "${IVF_FULL:-}" ]           && echo "      IVF_FULL_NPROBE: \"$IVF_FULL\"" )
$( [ -n "${IVF_NPROBE:-}" ]         && echo "      IVF_NPROBE: \"$IVF_NPROBE\"" )
$( [ -n "${IVF_BBOX_SEED:-}" ]      && echo "      IVF_BBOX_SEED: \"$IVF_BBOX_SEED\"" )
$( [ -n "${IVF_BBOX_VISIT_CAP:-}" ] && echo "      IVF_BBOX_VISIT_CAP: \"$IVF_BBOX_VISIT_CAP\"" )
  lb:
    image: $LB_TAG
    build: !reset null
EOF

echo ">> compose up"
wsl.exe -d "$WSL_DISTRO" -- bash -lc "cd $REPO_WSL && docker compose -f $COMPOSE_FILE -f $OVERRIDE_FILE down -v >/dev/null 2>&1 || true"
wsl.exe -d "$WSL_DISTRO" -- bash -lc "cd $REPO_WSL && docker compose -f $COMPOSE_FILE -f $OVERRIDE_FILE up -d >/dev/null"

echo ">> waiting for /ready"
ready=0
for i in $(seq 1 60); do
  code=$(wsl.exe -d "$WSL_DISTRO" -- bash -lc "curl -s -o /dev/null -w '%{http_code}' http://localhost:9999/ready" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    echo "ready after ${i}s"
    ready=1
    break
  fi
  sleep 1
done
if [ "$ready" != "1" ]; then
  echo "FAIL: server never became ready"
  wsl.exe -d "$WSL_DISTRO" -- bash -lc "cd $REPO_WSL && docker compose -f $COMPOSE_FILE -f $OVERRIDE_FILE logs --tail=30"
  wsl.exe -d "$WSL_DISTRO" -- bash -lc "cd $REPO_WSL && docker compose -f $COMPOSE_FILE -f $OVERRIDE_FILE down -v >/dev/null 2>&1 || true"
  rm -f "$OVERRIDE_FILE"
  exit 1
fi

echo ">> k6 run (120s ramp to 900 rps)"
wsl.exe -d "$WSL_DISTRO" -- bash -lc "set -e
cd $REPO_WSL
export K6_NO_USAGE_REPORT=true
k6 run test/test.js
" 2>&1 | tail -20 || true

wsl.exe -d "$WSL_DISTRO" -- bash -lc "cp $REPO_WSL/test/results.json $REPO_WSL/$OUT_DIR/$LABEL.json"

echo ">> compose down"
wsl.exe -d "$WSL_DISTRO" -- bash -lc "cd $REPO_WSL && docker compose -f $COMPOSE_FILE -f $OVERRIDE_FILE down -v >/dev/null 2>&1 || true"

rm -f "$OVERRIDE_FILE"

echo
echo "== result ($LABEL) =="
node -e '
const j = JSON.parse(require("fs").readFileSync("'"$OUT_DIR/$LABEL.json"'", "utf8"));
const s = j.scoring;
const b = s.breakdown;
const pp = (n) => (typeof n === "number" ? n.toFixed(2) : String(n));
console.log("p99           " + j.p99);
console.log("FP/FN/Err     " + b.false_positive_detections + "/" + b.false_negative_detections + "/" + b.http_errors);
console.log("TP/TN         " + b.true_positive_detections + "/" + b.true_negative_detections);
console.log("failure_rate  " + s.failure_rate);
console.log("E (weighted)  " + s.weighted_errors_E);
console.log("epsilon       " + s.error_rate_epsilon);
console.log("p99_score     " + pp(s.p99_score.value) + (s.p99_score.cut_triggered ? "  [CUT]" : ""));
console.log("det_score     " + pp(s.detection_score.value) + (s.detection_score.cut_triggered ? "  [CUT]" : ""));
console.log("              rate=" + pp(s.detection_score.rate_component) + "  penalty=" + pp(s.detection_score.absolute_penalty));
console.log("final_score   " + pp(s.final_score));
'
