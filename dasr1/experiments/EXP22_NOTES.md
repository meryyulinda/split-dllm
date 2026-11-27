# Exp22: several users on one cloud GPU

## Question
How many users can one cloud GPU serve, and how long does each answer take, when the model is
split (the cloud runs only layers k+1..28) versus cloud-only (the cloud runs every layer)?
Exp17 estimated the cloud work saved from one user at a time; here several users really share
one cloud GPU, so the cloud can work for one user while another user's step is on the network.

## Setup
- Fast-dLLM backend, Dream, bf16 rows, the first 16 GSM8K problems. dLLM-Cache is not used:
  its cache is one global object per process, so two users cannot share one model copy.
- Cloud (`exp22_cloud.py`, GPU 1): ONE model copy; every connection is served in its own
  thread and CUDA stream. No batching: each user's steps are separate model calls.
- Network (`src/net_relay.py --duplex`): every user gets their own two-way link.
  - `paper`: 100 Mbit/s, 20 ms round trip (the paper's standard setting).
  - `measured`: 89 Mbit/s, 112 ms: the Korea-to-lab path of Exp21, fitted to its 39 answers
    (network time = steps x round trip + bytes / throughput; median error 6%).
- Edge (`exp22_edge.py`, GPU 2): N users as threads, each with its own connection and CUDA
  stream, sharing one copy of the edge layers; edge steps slowed 3x (`--slowdown 3`), as in the
  paper.
- Configurations: cloud-only, split after layer 4, split after layer 20.
- Users: 1, 2, 4, 8, 12, 16, all starting together. Each user answers max(16 / N, 6) questions,
  user u starting at question u * 16 / N, and asks again as soon as it holds an answer.

## Run
```
nohup bash experiments/run_exp22.sh > exp22.log 2>&1 &          # waits for idle GPUs 1, 2; ~2 h
SMOKE=1 SKIP_IDLE_CHECK=1 bash experiments/run_exp22.sh > exp22_smoke.log 2>&1   # a few minutes
```
Re-running the same command resumes (finished configuration/link/users are skipped).

## What to read (`outputs/exp22_multiuser_<tag>/summary.csv`)
- `answers_per_hour`: each user's answers divided by that user's active time, summed over the
  users: the throughput of the one cloud GPU.
- `latency_s`, `latency_median_s`, `latency_p90_s`: time per answer as a user sees it.
- `same_as_one_user`: share of answers identical to the same question answered by one user
  alone (sharing the GPU must not change answers).
- `edge_s`, `wire_s`, `cloud_gen_s`: where the time goes.
- `gpu_pmon_exp22_<tag>.log`: per-process GPU use every second (also checks that no other job
  computed on GPUs 1 and 2).

Expected shape: cloud-only throughput is limited by the cloud GPU from few users on; split
throughput keeps rising with users while the cloud waits on the network, up to a limit set by
the cloud's share of the work (higher for split 20 than for split 4).
