"""User-space link emulator for Exp16: one direction of an edge-cloud link.

Runs as its own process (no sudo, no tc/netem). It accepts ONE TCP connection on
--listen, connects to --target and forwards the byte stream with

  * a bottleneck of --mbps (or a bandwidth trace, --trace): a chunk that arrives
    while the link is busy waits; its transmission takes
        (bytes + header_bytes * packets) * 8 / rate
    where packets = ceil(bytes / mss) (TCP/IP headers, 40 B per 1460-B segment by
    default; --header-bytes 0 turns this off);
  * a one-way propagation delay of --delay-ms (RTT/2 for a symmetric link) added
    after the transmission;
  * an unlimited FIFO queue (no loss, no reordering).

This is the link the analytical latency model assumes (fixed rate and delay, no
loss), so Exp16 checks the model's accounting -- the real payload, real sockets,
GPU<->CPU copies and framing -- rather than TCP congestion behaviour on a lossy
link (for that: Mahimahi or tc/netem on a machine we control).

Usage (started by src.net_wire.start_relay; can also be run by hand):
  python -m src.net_relay --listen 127.0.0.1:0 --target 127.0.0.1:5001 \
      --mbps 100 --delay-ms 10
The chosen listen port is printed as "PORT <n>" on stdout.

Trace file (--trace): CSV with a header line and columns time_s,mbps; the rate is
piecewise constant and the trace loops. A chunk uses the rate at its start time.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import socket
import sys
import threading
import time

CHUNK = 64 * 1024


class Rate:
    """Constant rate or a looping piecewise-constant trace (bits per second)."""

    def __init__(self, mbps=None, trace=None):
        self.points = None
        if trace:
            with open(trace, newline="") as f:
                rows = [(float(r["time_s"]), float(r["mbps"])) for r in csv.DictReader(f)]
            if not rows:
                raise ValueError(f"empty trace {trace}")
            rows.sort()
            self.points = rows
            self.period = rows[-1][0] if rows[-1][0] > 0 else 1.0
        elif mbps is None or mbps <= 0:
            raise ValueError("give --mbps > 0 or --trace")
        self.mbps = mbps
        self.t0 = time.perf_counter()

    def bps(self, t):
        if self.points is None:
            return self.mbps * 1e6
        rel = (t - self.t0) % self.period
        value = self.points[0][1]
        for ts, mbps in self.points:
            if ts > rel:
                break
            value = mbps
        return max(value, 1e-3) * 1e6


def wire_bytes(n, header_bytes, mss):
    if header_bytes <= 0:
        return n
    return n + header_bytes * math.ceil(n / mss)


def serve(listen, target, rate, delay_s, header_bytes=40, mss=1460, log=None):
    host, port = listen
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"PORT {srv.getsockname()[1]}", flush=True)
    src, _ = srv.accept()
    dst = socket.create_connection(target)
    for s in (src, dst):
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    queue = collections.deque()
    cond = threading.Condition()
    state = {"free": 0.0, "closed": False, "bytes": 0, "chunks": 0}

    def reader():
        try:
            while True:
                data = src.recv(CHUNK)
                if not data:
                    break
                now = time.perf_counter()
                start = max(now, state["free"])
                tx = wire_bytes(len(data), header_bytes, mss) * 8 / rate.bps(start)
                state["free"] = start + tx
                with cond:
                    queue.append((state["free"] + delay_s, data))
                    cond.notify()
                state["bytes"] += len(data)
                state["chunks"] += 1
        finally:
            with cond:
                state["closed"] = True
                cond.notify()

    def writer():
        while True:
            with cond:
                while not queue and not state["closed"]:
                    cond.wait()
                if not queue and state["closed"]:
                    break
                due, data = queue.popleft()
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            dst.sendall(data)
        dst.shutdown(socket.SHUT_WR)

    threads = [threading.Thread(target=reader, daemon=True),
               threading.Thread(target=writer, daemon=True)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for s in (src, dst, srv):
        s.close()
    if log:
        print(f"relay done: {state['bytes']} bytes in {state['chunks']} chunks", file=log,
              flush=True)


def _addr(text):
    host, port = text.rsplit(":", 1)
    return host, int(port)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--listen", type=_addr, required=True)
    ap.add_argument("--target", type=_addr, required=True)
    ap.add_argument("--mbps", type=float, default=None)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--delay-ms", type=float, default=0.0)
    ap.add_argument("--header-bytes", type=int, default=40)
    ap.add_argument("--mss", type=int, default=1460)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    serve(a.listen, a.target, Rate(a.mbps, a.trace), a.delay_ms / 1000.0,
          a.header_bytes, a.mss, log=sys.stderr if a.verbose else None)


if __name__ == "__main__":
    main()
