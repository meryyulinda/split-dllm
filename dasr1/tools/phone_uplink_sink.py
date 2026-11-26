"""Record a real uplink bandwidth trace from a phone (runs on the SERVER).

The phone pushes random bytes as fast as its uplink allows through SSH; this script
reads them from stdin and writes how many arrived in every bin as a bandwidth trace
for the network emulator (CSV time_s,mbps, last row = period), e.g. for Exp18.

On the phone (Termux, see tools/PHONE_NOTES.md):
  head -c 600M /dev/urandom | ssh -o Compression=no USER@SERVER \
    "cd ~/split-dllm/dasr1 && python3 tools/phone_uplink_sink.py outputs/phone_traces/lte_home.csv --seconds 120"

The trace is the throughput of one saturating TCP flow (with SSH encryption), i.e. a
little below what the radio link could carry, measured at the receiver.
"""
import argparse
import csv
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="CSV to write (time_s,mbps)")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--bin-ms", type=float, default=100.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    src = sys.stdin.buffer
    bins, t0, total = [], None, 0
    binw = a.bin_ms / 1000.0
    while True:
        chunk = src.read1(1 << 16)
        now = time.perf_counter()
        if not chunk:
            break
        if t0 is None:
            t0 = now
        k = int((now - t0) / binw)
        if now - t0 >= a.seconds:
            break
        while len(bins) <= k:
            bins.append(0)
        bins[k] += len(chunk)
        total += len(chunk)
    if not bins:
        sys.exit("no data received")
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "mbps"])
        for i, b in enumerate(bins):
            w.writerow([f"{i * binw:.3f}", f"{b * 8 / binw / 1e6:.4f}"])
        w.writerow([f"{len(bins) * binw:.3f}", f"{bins[0] * 8 / binw / 1e6:.4f}"])
    dur = len(bins) * binw
    mbps = sorted(b * 8 / binw / 1e6 for b in bins)
    print(f"{out}: {dur:.1f} s, {total / 1e6:.1f} MB, mean {total * 8 / dur / 1e6:.2f} Mbit/s, "
          f"p10 {mbps[len(mbps) // 10]:.2f}, median {mbps[len(mbps) // 2]:.2f}, "
          f"p90 {mbps[9 * len(mbps) // 10]:.2f} Mbit/s", file=sys.stderr)


if __name__ == "__main__":
    main()
