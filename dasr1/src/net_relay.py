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

Several users (Exp22, --duplex): the relay accepts any number of connections and gives
each its own two-way link: edge -> cloud at --mbps, cloud -> edge at --down-mbps
(default: the same), each direction with its own bottleneck and --delay-ms. Every
user has an access link of their own; the cloud's side is not a bottleneck.
  python -m src.net_relay --duplex --listen 127.0.0.1:6211 --target 127.0.0.1:6200 \
      --mbps 100 --delay-ms 10

Trace file (--trace): CSV with a header line and columns time_s,mbps; the rate is
piecewise constant, the last row's time is the period, and the trace loops. A chunk's
transmission is integrated over the trace segments it spans (zero-rate segments are
outages). --trace-offset starts the trace clock at that position (s) at the first byte.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import csv
import math
import socket
import sys
import threading
import time

CHUNK = 64 * 1024


class Rate:
    """Constant rate, or a looping piecewise-constant bandwidth trace.

    Trace CSV: time_s,mbps (sorted); the value of a row holds until the next row,
    and the last row's time is the period after which the trace repeats. The
    trace clock starts at the first byte (start()) plus `offset_s`, so every run
    of an experiment can replay the same part of the trace.
    """

    def __init__(self, mbps=None, trace=None, offset_s=0.0):
        self.points = None
        if trace:
            with open(trace, newline="") as f:
                rows = [(float(r["time_s"]), float(r["mbps"])) for r in csv.DictReader(f)]
            if len(rows) < 2:
                raise ValueError(f"trace {trace} needs at least two rows")
            rows.sort()
            self.times = [t for t, _ in rows]
            self.points = [max(m, 0.0) * 1e6 for _, m in rows]     # bits/s per segment
            self.period = self.times[-1]
            if self.period <= 0 or not any(self.points[:-1]):
                raise ValueError(f"trace {trace}: empty period or zero bandwidth throughout")
        elif mbps is None or mbps <= 0:
            raise ValueError("give --mbps > 0 or --trace")
        self.mbps = mbps
        self.offset = float(offset_s)
        self.t0 = None

    def start(self, now):
        if self.t0 is None:
            self.t0 = now

    def finish(self, start, bits):
        """Time at which `bits` sent from `start` (perf_counter s) have left the link."""
        if self.points is None:
            return start + bits / (self.mbps * 1e6)
        rel = (start - self.t0 + self.offset) % self.period
        t = start
        i = min(max(0, bisect.bisect_right(self.times, rel) - 1), len(self.times) - 2)
        guard = 0
        while bits > 0:
            seg_end = self.times[i + 1] if i + 1 < len(self.times) else self.period
            span = seg_end - rel
            cap = self.points[i] * span
            if cap >= bits:
                return t + bits / self.points[i]
            bits -= cap
            t += span
            rel = seg_end
            i += 1
            if i >= len(self.times) - 1:     # wrap around
                i, rel = 0, 0.0
            guard += 1
            if guard > 10_000_000:
                raise RuntimeError("trace has no capacity")
        return t


def wire_bytes(n, header_bytes, mss):
    if header_bytes <= 0:
        return n
    return n + header_bytes * math.ceil(n / mss)


def _pump(src, dst, rate, delay_s, header_bytes, mss, state):
    """Forward src -> dst through one link direction; returns the two threads
    (reader: bottleneck and delay; writer: releases each chunk when it is due)."""
    queue = collections.deque()
    cond = threading.Condition()
    state.update({"free": 0.0, "closed": False, "bytes": 0, "chunks": 0})

    def reader():
        try:
            while True:
                data = src.recv(CHUNK)
                if not data:
                    break
                now = time.perf_counter()
                rate.start(now)
                start = max(now, state["free"])
                state["free"] = rate.finish(start, wire_bytes(len(data), header_bytes, mss) * 8)
                with cond:
                    queue.append((state["free"] + delay_s, data))
                    cond.notify()
                state["bytes"] += len(data)
                state["chunks"] += 1
        except OSError:
            pass
        finally:
            with cond:
                state["closed"] = True
                cond.notify()

    def writer():
        try:
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
        except OSError:
            pass

    return [threading.Thread(target=reader, daemon=True),
            threading.Thread(target=writer, daemon=True)]


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
    state = {}
    threads = _pump(src, dst, rate, delay_s, header_bytes, mss, state)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for s in (src, dst, srv):
        s.close()
    if log:
        print(f"relay done: {state['bytes']} bytes in {state['chunks']} chunks", file=log,
              flush=True)


def serve_duplex(listen, target, make_up, make_down, delay_s, header_bytes=40, mss=1460,
                 log=None):
    """Accept connections forever; each gets its own two-way link (Exp22: one per user).
    make_up / make_down return a fresh Rate for each connection."""
    host, port = listen
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(64)
    print(f"PORT {srv.getsockname()[1]}", flush=True)

    def handle(src, peer):
        try:
            dst = socket.create_connection(target)
        except OSError as exc:
            if log:
                print(f"relay: cannot reach {target}: {exc!r}", file=log, flush=True)
            src.close()
            return
        for s in (src, dst):
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        up, down = {}, {}
        threads = (_pump(src, dst, make_up(), delay_s, header_bytes, mss, up)
                   + _pump(dst, src, make_down(), delay_s, header_bytes, mss, down))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for s in (src, dst):
            s.close()
        if log:
            print(f"relay {peer}: up {up['bytes']} B, down {down['bytes']} B", file=log,
                  flush=True)

    while True:
        src, peer = srv.accept()
        threading.Thread(target=handle, args=(src, peer), daemon=True).start()


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
    ap.add_argument("--trace-offset", type=float, default=0.0, help="start position in the trace (s)")
    ap.add_argument("--header-bytes", type=int, default=40)
    ap.add_argument("--mss", type=int, default=1460)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--duplex", action="store_true",
                    help="many connections, each with its own two-way link (Exp22)")
    ap.add_argument("--down-mbps", type=float, default=None,
                    help="with --duplex: cloud -> edge rate (default: --mbps)")
    a = ap.parse_args(argv)
    if a.duplex:
        down = a.down_mbps or a.mbps
        serve_duplex(a.listen, a.target, lambda: Rate(a.mbps, a.trace, a.trace_offset),
                     lambda: Rate(down), a.delay_ms / 1000.0, a.header_bytes, a.mss,
                     log=sys.stderr if a.verbose else None)
        return
    serve(a.listen, a.target, Rate(a.mbps, a.trace, a.trace_offset), a.delay_ms / 1000.0,
          a.header_bytes, a.mss, log=sys.stderr if a.verbose else None)


if __name__ == "__main__":
    main()
