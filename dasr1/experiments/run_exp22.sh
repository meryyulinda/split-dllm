#!/usr/bin/env bash
# Exp22: several users on one cloud GPU (Fast-dLLM backend, Dream, GSM8K, bf16 rows).
# Runs in the background and waits until both GPUs are idle (checked every 5 min):
#   nohup bash experiments/run_exp22.sh > exp22.log 2>&1 &
#   GPU_C=1 GPU_E=2 ...     cloud GPU, edge GPU (default 1 and 2)
#   SMOKE=1 ...             2 configurations, 1 link, 1-2 users, 2 questions (a few minutes)
#   SKIP_IDLE_CHECK=1 ...   start at once
# For every configuration (cloud-only, split after layer 4, split after layer 20) it starts
# one cloud program (exp22_cloud.py, one model copy for all users), one network emulator
# per link (src/net_relay.py --duplex: every user gets its own link), and the edge
# program (exp22_edge.py: 1, 2, 4, 8, 12, 16 users at once), then stops them.
# Re-running the same command resumes: finished (configuration, link, users) are skipped.
# Output: outputs/exp22_multiuser_<TAG>/, logs exp22_cloud_<config>.log, gpu_pmon_exp22.log
set -uo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU_C="${GPU_C:-1}"; GPU_E="${GPU_E:-2}"
TAG="${TAG:-lab}"
CONFIGS="${CONFIGS:-cloud_only:0 split4:4 split20:20}"
LINKS="${LINKS:-paper:100:20 measured:97:155}"     # name:Mbit/s:RTT ms (measured = Exp21 Korea path)
USERS="${USERS:-1,2,4,8,12,16}"
POOL="${POOL:-16}"; MINQ="${MINQ:-6}"
SLOWDOWN="${SLOWDOWN:-3}"                           # edge 3x slower than an A6000, as in the paper
WAIT_SECONDS="${WAIT_SECONDS:-300}"
CLOUD_PORT=6200
if [ "${SMOKE:-0}" = "1" ]; then
  CONFIGS="cloud_only:0 split4:4"; LINKS="paper:100:20"; USERS="1,2"; POOL=2; MINQ=1
  TAG="${TAG}_smoke"
fi

if [ "${SKIP_IDLE_CHECK:-0}" != "1" ]; then
  while true; do
    busy=""
    for gpu in $GPU_C $GPU_E; do
      max_util=0
      for _ in 1 2 3 4 5; do
        u=$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
        [ "$u" -gt "$max_util" ] && max_util=$u
        sleep 1
      done
      [ "$max_util" -gt 10 ] && busy="$busy GPU $gpu (${max_util}%)"
    done
    if [ -z "$busy" ]; then echo "$(date '+%F %T') GPUs $GPU_C,$GPU_E idle, starting."; break; fi
    echo "$(date '+%F %T') busy:$busy; checking again in ${WAIT_SECONDS} s ..."
    sleep "$WAIT_SECONDS"
  done
fi

nvidia-smi pmon -s um -d 1 -o T > "gpu_pmon_exp22_${TAG}.log" 2>&1 &
PMON=$!
cleanup() { kill ${RELAYS[@]+"${RELAYS[@]}"} ${CLOUD:+"$CLOUD"} "$PMON" 2>/dev/null; }
trap cleanup EXIT
RELAYS=()

for item in $CONFIGS; do
  name="${item%%:*}"; k="${item##*:}"
  log="exp22_cloud_${name}_${TAG}.log"
  echo "$(date '+%F %T') === $name (split $k): starting the cloud program on GPU $GPU_C"
  CUDA_VISIBLE_DEVICES=$GPU_C python experiments/exp22_cloud.py --split "$k" --port $CLOUD_PORT > "$log" 2>&1 &
  CLOUD=$!
  ready=0
  for _ in $(seq 1 180); do       # up to 30 min to load
    if grep -q "listening" "$log"; then ready=1; break; fi
    if ! kill -0 "$CLOUD" 2>/dev/null; then break; fi
    sleep 10
  done
  if [ "$ready" != "1" ]; then
    echo "$(date '+%F %T') cloud program for $name did not start; see $log"; tail -5 "$log"
    kill "$CLOUD" 2>/dev/null; continue
  fi
  RELAYS=(); spec=""; i=1
  for l in $LINKS; do
    lname="${l%%:*}"; rest="${l#*:}"; mbps="${rest%%:*}"; rtt="${rest##*:}"
    port=$((CLOUD_PORT + 10 + i))
    half=$(python -c "print($rtt / 2)")
    python -m src.net_relay --duplex --listen 127.0.0.1:$port --target 127.0.0.1:$CLOUD_PORT \
      --mbps "$mbps" --delay-ms "$half" > "exp22_relay_${lname}.log" 2>&1 &
    RELAYS+=($!)
    spec="${spec:+$spec,}$lname=127.0.0.1:$port:$mbps:$rtt"
    i=$((i + 1))
  done
  sleep 3
  CUDA_VISIBLE_DEVICES=$GPU_E python experiments/exp22_edge.py --config "$name" --split "$k" \
    --links "$spec" --users "$USERS" --pool "$POOL" --min-q "$MINQ" --slowdown "$SLOWDOWN" \
    --tag "$TAG" || echo "$(date '+%F %T') edge program for $name failed (re-run this script to resume)"
  kill ${RELAYS[@]+"${RELAYS[@]}"} "$CLOUD" 2>/dev/null
  wait "$CLOUD" 2>/dev/null
  RELAYS=(); CLOUD=""
  sleep 5
done
echo "$(date '+%F %T') Exp22 completed. Output: outputs/exp22_multiuser_${TAG}/"
