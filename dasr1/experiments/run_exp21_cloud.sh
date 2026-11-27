#!/usr/bin/env bash
# Exp21 cloud side (lab server): starts the four cloud programs in the background, after
# GPUs 1 and 2 are idle (checked every 5 min), and keeps them running until stopped:
#   port 6001  dLLM-Cache backend, split after layer 4         (GPU 1)
#   port 6002  dLLM-Cache backend, cloud-only (split 0)         (GPU 1)
#   port 6003  Fast-dLLM backend, split after layer 4           (GPU 2)
#   port 6004  Fast-dLLM backend, cloud-only (split 0)          (GPU 2)
# They listen on 127.0.0.1 only. Run from dasr1/:
#   nohup bash experiments/run_exp21_cloud.sh > exp21_cloud.log 2>&1 &
#   GPU_A=1 GPU_B=2 ...       other GPUs
# Stop them afterwards:  pkill -f exp21_cloud.py ; pkill -f "nvidia-smi pmon -s um -d 5 -o T"
# A remote edge (vast.ai) reaches them through a reverse SSH tunnel opened FROM the lab server:
#   ssh -i ~/.ssh/id_ed25519_meryvast -p <PORT> root@<ADDRESS> -N -o ServerAliveInterval=30 \
#       -o ExitOnForwardFailure=yes -R 6001:127.0.0.1:6001 -R 6002:127.0.0.1:6002 \
#       -R 6003:127.0.0.1:6003 -R 6004:127.0.0.1:6004
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU_A="${GPU_A:-1}"; GPU_B="${GPU_B:-2}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
SKIP_IDLE_CHECK="${SKIP_IDLE_CHECK:-0}"

if [ "$SKIP_IDLE_CHECK" != "1" ]; then
  while true; do
    busy=""
    for gpu in $GPU_A $GPU_B; do
      max_util=0
      for _ in 1 2 3 4 5; do
        u=$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
        [ "$u" -gt "$max_util" ] && max_util=$u
        sleep 1
      done
      [ "$max_util" -gt 10 ] && busy="$busy GPU $gpu (${max_util}%)"
    done
    if [ -z "$busy" ]; then echo "$(date '+%F %T') GPUs $GPU_A,$GPU_B idle, starting."; break; fi
    echo "$(date '+%F %T') busy:$busy; checking again in ${WAIT_SECONDS} s ..."
    sleep "$WAIT_SECONDS"
  done
fi

nvidia-smi pmon -s um -d 5 -o T > gpu_pmon_exp21.log 2>&1 &
start() {  # $1 gpu, $2 backend, $3 split, $4 port
  CUDA_VISIBLE_DEVICES=$1 nohup python experiments/exp21_cloud.py --backend "$2" --split "$3" \
    --port "$4" > "exp21_cloud_$2_k$3.log" 2>&1 &
  echo "started $2 split=$3 on GPU $1, port $4 (log exp21_cloud_$2_k$3.log)"
}
start "$GPU_A" dllm_cache 4 6001
start "$GPU_A" dllm_cache 0 6002
start "$GPU_B" fastdllm 4 6003
start "$GPU_B" fastdllm 0 6004
echo "$(date '+%F %T') all four cloud programs started; they print 'listening' once loaded."
wait
