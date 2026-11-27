# Exp21: DASR as two programs, on two real machines

## What changes
Until Exp20 the edge and cloud halves ran in one Python program on the lab server. Exp21
splits DASR into:
- an **edge program** (`exp21_edge.py`), with the tokenizer, the embedding, Dream's layers
  1..4, their caches and the codec;
- a **cloud program** (`exp21_cloud.py`), with layers 5..28, the LM head, the sampler and the
  cloud caches.

They talk over TCP (`src/remote_split.py`). Per denoising step there is one round trip: the
cloud sends the committed tokens plus what to compute, and the edge returns the boundary
rows.

The cloud runs the unchanged samplers. Its embedding is replaced by one that returns the rows
received from the edge, so the cloud's decisions (refresh, blocks, codec anchors) are made as
before and sent along.

The edge reproduces what layers 1..4 did:
- **dLLM-Cache:** the official hook, with the cloud's refresh flags.
- **Fast-dLLM:** its own key/value cache for its layers.

The **cloud-only reference** is the same cloud program with `--split 0`: the edge sends only the
prompt and receives the answer over the same network path.

## Step 1: check on the lab server (both programs there; no rental)
```
nohup bash experiments/run_exp21_cloud.sh > exp21_cloud.log 2>&1 &                     # waits for idle GPUs 1,2
TAG=lab_a6000 GPU=2 nohup bash experiments/run_exp21_edge.sh > exp21_edge_lab.log 2>&1 &
```
- The edge then runs on an A6000 like the earlier experiments, so every answer must equal the
  single-process run of Exp16 (`same_as_single_process`, INT8 included).
- Add `SMOKE=1` for a 2-question check first.
- Stop the cloud programs with `pkill -f exp21_cloud.py` when done.

## Step 2: real deployment (edge = rented RTX 3060 in Taiwan)
1. **Rent** the RTX 3060 (SSH launch mode, about 40 GB disk). On it:
   ```
   curl -O https://raw.githubusercontent.com/meryyulinda/split-dllm/main/dasr1/experiments/setup_exp21_edge.sh
   bash setup_exp21_edge.sh
   ```
2. **On the lab server,** with the cloud programs running (step 1's first command), open the
   tunnel from the lab server to the rented machine. Port and address come from Vast's Connect
   button.
   ```
   nohup ssh -i ~/.ssh/id_ed25519_meryvast -p <PORT> root@<ADDRESS> -N -o ServerAliveInterval=30 \
     -o ExitOnForwardFailure=yes -R 6001:127.0.0.1:6001 -R 6002:127.0.0.1:6002 \
     -R 6003:127.0.0.1:6003 -R 6004:127.0.0.1:6004 > exp21_tunnel.log 2>&1 &
   ```
   This makes the cloud programs reachable on the rented machine's own ports 6001–6004. The lab
   server opens no port to the internet.
3. **On the rented machine:**
   ```
   cd ~/work/split-dllm/dasr1
   TAG=vast_3060_tw nohup bash experiments/run_exp21_edge.sh > exp21_edge.log 2>&1 &
   ```
   It downloads only the checkpoint files that hold the embedding and layers 1..4 (a few GB).
   It needs about 4 GB of GPU memory.
4. **When it is done,** copy `outputs/exp21_deploy_vast_3060_tw/` to the lab server and push:
   ```
   scp -i ~/.ssh/id_ed25519_meryvast -P <PORT> -r root@<ADDRESS>:~/work/split-dllm/dasr1/outputs/exp21_deploy_vast_3060_tw outputs/
   ```
   Then destroy the instance.

## What it measures (summary.csv, per server and codec)
- `latency_s`: the real time per answer at the edge, from sending the prompt until it holds the
  answer.
- Its parts: `edge_s` (edge GPU), `cloud_s` (cloud layers), `wire_s` (network and copies, seen
  by the cloud).
- `steps`, `up_mb`, and `acc`.
- `same_as_single_process`: answers identical to Exp16's single-process runs. Expect all on the
  lab check. On the 3060 some may differ, because a different GPU computes layers 1..4 with
  slightly different rounding.

The cloud programs also log every answer's per-step statistics to
`outputs/exp21_cloud_logs/<backend>_k<split>.jsonl`.

## Next (Exp22): many edge devices on one cloud GPU
The cloud program serves one connection at a time for now. The multi-user run adds concurrent
sessions on one cloud model, plus several edge processes, to measure answers per hour and
latency against the number of users, for the split and for cloud-only inference.
