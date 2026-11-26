# Using an Android phone for the real edge–cloud setup

With only a phone (no laptop GPU or Jetson), there are two realistic measurements:

1. **Real uplink traces from the phone to our server.** These are replayed in the network
   emulator (Exp18), instead of or next to the Mahimahi LTE traces. This is cheap and
   useful.
2. **How fast the phone's CPU could run the edge part.** This measures the edge
   slowdown s that the paper currently assumes (3x).

Running the actual edge half of Dream-7B on the phone would need a mobile inference
runtime (e.g. MLC-LLM or ExecuTorch) with Dream's layers and the dLLM-Cache /
Fast-dLLM hooks ported. That is a separate engineering project, and measurement 2 shows
whether it is worth it.

## Setup (once)
- Install **Termux** from F-Droid (the Play Store version is outdated).
- In Termux:
  ```
  pkg update && pkg install python python-numpy openssh curl
  ```
- The phone must reach the server over SSH. If you normally use the campus VPN, turn
  it on in the phone too. Check with `ssh USER@SERVER echo ok`.

## 1. Uplink traces (~2.5 min each)
From the phone, for each network you want: 4G/5G at home, at the lab, Wi-Fi, and so on.
```
head -c 800M /dev/urandom | ssh -o Compression=no USER@SERVER \
  "cd ~/split-dllm/dasr1 && python3 tools/phone_uplink_sink.py outputs/phone_traces/lte_lab.csv --seconds 120"
ping -c 30 SERVER        # note the average RTT
```
- The server prints the mean and percentiles.
- If the phone's uplink is faster than ~50 Mbit/s, use more than 800M so the upload
  lasts the whole 120 s.
- A 'broken pipe' message at the end is expected: the server stops after 120 s.
- Replay a trace in Exp18 by adding it to `EXP18_TRACES`, e.g.
  `{"name": "phone_lte_lab", "csv": "outputs/phone_traces/lte_lab.csv", "n": 10}`,
  and the measured RTT as `EXP18_RTT_MS`.

## 2. Edge compute (~2–5 min)
```
curl -O https://raw.githubusercontent.com/meryyulinda/split-dllm/main/dasr1/tools/phone_edge_bench.py
python phone_edge_bench.py
```
- Send back the last line (it starts with `JSON`).
- It times the matrix work of Dream's first 4 layers on the phone's CPU (float32, numpy).
  A 2-core server VM needs about 7 s per step at 360 rows, i.e. about 860x slower than
  the A6000. A phone CPU will likely be in the hundreds as well.
- Every denoising step recomputes the rows the edge handles: 32 rows in-block with
  Fast-dLLM, all rows otherwise. So a phone CPU is far too slow for a 7B dLLM.
- The measurement turns the paper's "edge 3x slower" assumption into a known range.
