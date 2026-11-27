#!/usr/bin/env bash
# Exp21 watcher (lab server, from dasr1/): waits for the Vast edge run to finish, restarts the
# edge if it crashed (it resumes from runs.csv), copies the results back, stops the tunnel and
# the cloud programs, and destroys the Vast instance. After MAX_HOURS it copies what exists
# and destroys the instance anyway.
#   PORT=<port> ADDR=<ip> ID=<instance id> nohup bash exp21_finish.sh > exp21_finish.log 2>&1 &
set -uo pipefail
: "${PORT:?set PORT}" "${ADDR:?set ADDR}" "${ID:?set ID}"
TAG="${TAG:-vast_3060_kr}"
MAX_HOURS="${MAX_HOURS:-7}"
KEY="$HOME/.ssh/id_ed25519_meryvast"
REMOTE="~/work/split-dllm/dasr1"
VASTAI="$(command -v vastai || echo "$HOME/.local/bin/vastai")"
SSH=(ssh -i "$KEY" -p "$PORT" -o BatchMode=yes -o ConnectTimeout=20 "root@$ADDR")
log() { echo "$(date '+%F %T') $*"; }

"$VASTAI" show instances > /dev/null || { log "vastai CLI not working, stopping"; exit 1; }
deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
restarts=0
while true; do
  if "${SSH[@]}" "grep -q 'Exp21 edge completed' $REMOTE/exp21_edge.log"; then
    log "edge run completed"; break
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "reached ${MAX_HOURS} h, copying what exists"; break
  fi
  running=$("${SSH[@]}" "pgrep -f '[r]un_exp21_edge' > /dev/null && echo yes || echo no")
  if [ "$running" = "no" ]; then
    if [ "$restarts" -ge 5 ]; then log "edge crashed 5 times, copying what exists"; break; fi
    restarts=$((restarts + 1))
    log "edge not running, restart $restarts/5 (resumes from runs.csv)"
    "${SSH[@]}" "cd $REMOTE && TAG=$TAG nohup setsid bash experiments/run_exp21_edge.sh >> exp21_edge.log 2>&1 < /dev/null &"
  else
    log "running: $("${SSH[@]}" "grep -c 'same=' $REMOTE/exp21_edge.log") answers done"
  fi
  sleep 300
done

copied=no
for try in 1 2 3 4 5; do
  if scp -i "$KEY" -P "$PORT" -o BatchMode=yes -r "root@$ADDR:$REMOTE/outputs/exp21_deploy_$TAG" outputs/ &&
     scp -i "$KEY" -P "$PORT" -o BatchMode=yes "root@$ADDR:$REMOTE/exp21_edge.log" "exp21_edge_$TAG.log"; then
    copied=yes; break
  fi
  log "copy failed (try $try/5)"; sleep 60
done
log "copied: $copied ($(ls outputs/exp21_deploy_$TAG 2>/dev/null | tr '\n' ' '))"

pkill -f "while true; do ssh" ; pkill -f "^ssh -i .*R 6001"
pkill -f exp21_cloud.py ; pkill -f "nvidia-smi pmon -s um -d 5 -o T"
log "tunnel and cloud programs stopped"

if [ "$copied" = "yes" ] || [ "$(date +%s)" -gt "$deadline" ]; then
  yes | "$VASTAI" destroy instance "$ID"
  sleep 30
  log "instances left on the account:"; "$VASTAI" show instances
else
  log "results NOT copied, instance $ID kept; copy and destroy it by hand"
fi
log "done"