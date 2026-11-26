"""Real edge-cloud link for Exp16: framed messages over TCP through src.net_relay.

The two halves of the model stay in one Python process (same model code and
samplers as Exp11 / Exp13, so outputs are directly comparable), but every message
between them now travels as bytes over a real TCP connection through a separate
relay process that imposes the bandwidth and the delay:

  uplink   edge -> relay(up)   -> cloud : the boundary rows, wire format below
  downlink cloud -> relay(down) -> edge : the tokens committed since the last step

The cloud layers read ONLY what was decoded from the received bytes, and the edge
embeds ONLY the tokens it has received, so a wrong wire format would change the
answers (Exp16 checks them against a run without the network).

Wire format (sizes match the byte counts of src.split_runtime / the latency model)
  uplink header, 16 B   <BBHIQ  type, flags, step mod 2^16, first row, body bytes
    type 1 bf16 rows    body = rows x d x 2 B (native bf16)
    type 2 INT8 delta   body = rows x d x 1 B (int8) + rows x 2 B (fp16 scale);
                        the cloud adds q * scale to its own copy of the reference
  downlink header, 8 B  <BBHI   type, flags, step mod 2^16, count
    body = count x (uint16 position + uint32 token id) = 6 B per token
  type 9 = ping (empty body), used to measure the real round-trip time.

Timing: the edge GPU is synchronized before a message is built and the cloud GPU
before a reply is built, so the recorded wire times (up_wire_ms, down_wire_ms)
contain no GPU compute: GPU->CPU copy, framing, socket, relay (rate + delay),
receive, CPU->GPU copy and decoding. edge_slowdown s > 1 emulates a slower edge by
sleeping (s - 1) x the measured edge compute of the step before sending.

Transport (Channel, start_relay) has no torch dependency and is tested on its own.
"""
from __future__ import annotations

import queue
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

DASR_ROOT = Path(__file__).resolve().parents[1]

UP_FMT, DOWN_FMT = "<BBHIQ", "<BBHI"
UP_HDR, DOWN_HDR = struct.calcsize(UP_FMT), struct.calcsize(DOWN_FMT)   # 16, 8
UP_BF16, UP_INT8, PING = 1, 2, 9
DOWN_COMMITS = 1
TIMEOUT_S = 900.0


# --------------------------------------------------------------------------- transport
def start_relay(target_port, mbps=None, delay_ms=0.0, trace=None, header_bytes=40, mss=1460,
                trace_offset=0.0):
    """Start src.net_relay as a separate process; return (process, listen port)."""
    cmd = [sys.executable, "-m", "src.net_relay", "--listen", "127.0.0.1:0",
           "--target", f"127.0.0.1:{target_port}", "--delay-ms", str(delay_ms),
           "--header-bytes", str(header_bytes), "--mss", str(mss)]
    cmd += (["--trace", str(Path(trace).resolve()), "--trace-offset", str(trace_offset)]
            if trace else ["--mbps", str(mbps)])
    proc = subprocess.Popen(cmd, cwd=str(DASR_ROOT), stdout=subprocess.PIPE, text=True)
    line = proc.stdout.readline().strip()
    if not line.startswith("PORT "):
        proc.kill()
        raise RuntimeError(f"relay did not start (got {line!r})")
    return proc, int(line.split()[1])


def _recv_exact(sock, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise EOFError("connection closed")
        got += k
    return buf


class Channel:
    """One direction: sender socket -> relay process -> receiver socket + thread.

    hdr_fmt: struct format of the fixed header; body_len(header_tuple) -> bytes.
    The receiver thread reads whole frames into a queue, so a large send never
    deadlocks against an unread receive buffer in the same process.
    """

    def __init__(self, hdr_fmt, body_len, *, mbps=None, delay_ms=0.0, trace=None,
                 header_bytes=40, mss=1460, trace_offset=0.0):
        self.fmt, self.hdr, self.body_len = hdr_fmt, struct.calcsize(hdr_fmt), body_len
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(60)
        self.relay, port = start_relay(listener.getsockname()[1], mbps, delay_ms, trace,
                                       header_bytes, mss, trace_offset)
        self.tx = socket.create_connection(("127.0.0.1", port), timeout=60)
        self.rx, _ = listener.accept()
        listener.close()
        for s in (self.tx, self.rx):
            s.settimeout(None)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
                s.setsockopt(socket.SOL_SOCKET, opt, 4 << 20)
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self):
        try:
            while True:
                head = struct.unpack(self.fmt, _recv_exact(self.rx, self.hdr))
                n = self.body_len(head)
                body = _recv_exact(self.rx, n) if n else bytearray()
                self.q.put((head, body))
        except (EOFError, OSError) as exc:
            self.q.put(exc)

    def send(self, header, *bodies):
        """Send one frame; numpy bodies go out without an extra copy. Returns its bytes."""
        n = len(header)
        self.tx.sendall(header)
        for b in bodies:
            if isinstance(b, np.ndarray):
                b = np.ascontiguousarray(b).reshape(-1).view(np.uint8)
            mv = memoryview(b)
            if mv.nbytes:
                self.tx.sendall(mv)
                n += mv.nbytes
        return n

    def recv(self, timeout=TIMEOUT_S):
        item = self.q.get(timeout=timeout)
        if isinstance(item, BaseException):
            raise RuntimeError(f"link closed: {item!r}")
        return item

    def close(self):
        try:
            self.tx.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        self.thread.join(timeout=30)
        for s in (self.tx, self.rx):
            s.close()
        try:
            self.relay.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.relay.kill()


def up_body_len(head):
    return int(head[4])


def down_body_len(head):
    return 6 * int(head[3])


class Link:
    """The two directions of an edge-cloud link (both through their own relay)."""

    def __init__(self, mbps=None, rtt_ms=0.0, trace=None, header_bytes=40, mss=1460,
                 trace_offset=0.0, down_trace=None):
        """down_trace: bandwidth trace of the downlink (default: the uplink's trace)."""
        kw = dict(mbps=mbps, delay_ms=rtt_ms / 2.0, header_bytes=header_bytes, mss=mss,
                  trace_offset=trace_offset)
        self.up = Channel(UP_FMT, up_body_len, trace=trace, **kw)
        self.down = Channel(DOWN_FMT, down_body_len, trace=down_trace or trace, **kw)

    def ping(self, n=5):
        """Real round-trip times (ms) of empty frames: up, then down."""
        out = []
        for i in range(n):
            t0 = time.perf_counter()
            self.up.send(struct.pack(UP_FMT, PING, 0, i, 0, 0))
            self.up.recv()
            self.down.send(struct.pack(DOWN_FMT, PING, 0, i, 0))
            self.down.recv()
            out.append((time.perf_counter() - t0) * 1e3)
        return out

    def close(self):
        self.up.close()
        self.down.close()


# --------------------------------------------------------------------------- tensors
def _sync(device):
    import torch
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class Wire:
    """Encodes / decodes the split's messages and sends them over a Link.

    Attach to a SplitLink or KVBlockLink as `link.wire = Wire(...)`; the links call
      reset()                       at begin()
      downlink(ids, lo, step)       at every step start (returns ids on the edge)
      uplink(step, lo, ev, ...)     at the boundary (returns the decoded rows on the cloud)
    and the experiment calls final(full_ids) after the generation (last commits).
    """

    def __init__(self, link, edge, cloud, edge_slowdown=1.0, check=True):
        self.link, self.edge, self.cloud = link, edge, cloud
        self.slow, self.check = float(edge_slowdown), bool(check)
        self.reset()

    def reset(self):
        self.record = None      # cloud: the tokens the edge holds (what was sent)
        self.edge_view = None   # edge: the tokens it holds (CPU, int64)
        self.down_mismatch = 0

    # ---------------------------------------------------------------- downlink
    def _send_commits(self, ids, lo, step):
        import torch
        cur = self.record[:, lo:lo + ids.shape[1]]
        idx = torch.nonzero(ids[0] != cur[0]).squeeze(1)
        vals = ids[0, idx]
        self.record[:, lo:lo + ids.shape[1]] = ids
        pos = (idx + lo).cpu().numpy().astype("<u2")
        tok = vals.cpu().numpy().astype("<u4")
        sent = self.link.down.send(struct.pack(DOWN_FMT, DOWN_COMMITS, 0, step & 0xFFFF, len(pos)),
                                   pos, tok)
        head, body = self.link.down.recv()                       # ---- edge side
        cnt = int(head[3])
        p = np.frombuffer(body, dtype="<u2", count=cnt).astype(np.int64)
        t = np.frombuffer(body, dtype="<u4", count=cnt, offset=2 * cnt).astype(np.int64)
        self.edge_view[0, torch.from_numpy(p)] = torch.from_numpy(t)
        return sent

    def downlink(self, ids, lo, step):
        """ids: the token rows of this forward (cloud tensor (1, m)) starting at row lo."""
        if self.record is None:  # first forward: the edge built the canvas itself
            if lo != 0:
                raise RuntimeError("the first forward must start at row 0")
            self.record = ids.detach().clone()
            self.edge_view = ids.detach().cpu().clone()
            return ids.to(self.edge), {"down_wire_ms": 0.0, "wire_down_bytes": 0}
        _sync(ids.device)
        t0 = time.perf_counter()
        sent = self._send_commits(ids, lo, step)
        ids_edge = self.edge_view[:, lo:lo + ids.shape[1]].to(self.edge)
        ms = (time.perf_counter() - t0) * 1e3
        if self.check and not bool((self.edge_view[:, lo:lo + ids.shape[1]] == ids.cpu()).all()):
            self.down_mismatch += 1
        return ids_edge, {"down_wire_ms": ms, "wire_down_bytes": sent}

    def final(self, full_ids):
        """Send the last step's commits so the edge holds the answer; returns info.

        Runs in inference mode: Dream's sampler creates the token records inside
        torch.inference_mode(), and such tensors cannot be updated outside it."""
        import torch
        _sync(full_ids.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            sent = self._send_commits(full_ids, 0, 0)
        ms = (time.perf_counter() - t0) * 1e3
        ok = bool((self.edge_view == full_ids.cpu()).all())
        return {"final_down_ms": ms, "final_down_bytes": sent, "edge_has_answer": ok,
                "down_mismatch_steps": self.down_mismatch}

    # ---------------------------------------------------------------- uplink
    def uplink(self, step, lo, ev, raw=None, q=None, scale=None, base=None):
        """raw: (1, r, d) rows sent natively; or q (1, r, d) rounded deltas + scale (1, r, 1)
        and base = the cloud's copy of the reference rows. Returns (rows on cloud, info)."""
        import torch
        _sync(self.edge)
        slept = 0.0
        if self.slow > 1.0 and ev is not None:
            slept = (self.slow - 1.0) * ev["edge_start"].elapsed_time(ev["encode_end"])
            time.sleep(slept / 1e3)
        t0 = time.perf_counter()
        if raw is not None:
            if raw.dtype != torch.bfloat16:
                raise TypeError(f"the bf16 wire format needs bf16 rows, not {raw.dtype}")
            r, d = int(raw.shape[1]), int(raw.shape[2])
            body = raw.contiguous().view(torch.int16).cpu().numpy()   # bf16 bits
            sent = self.link.up.send(struct.pack(UP_FMT, UP_BF16, 0, step & 0xFFFF, lo,
                                                 body.nbytes), body)
            dtype = raw.dtype
        else:
            r, d = int(q.shape[1]), int(q.shape[2])
            qb = q.to(torch.int8).contiguous().cpu().numpy()
            sb = scale.reshape(-1).half().contiguous().cpu().numpy()
            sent = self.link.up.send(struct.pack(UP_FMT, UP_INT8, 0, step & 0xFFFF, lo,
                                                 qb.nbytes + sb.nbytes), qb, sb)
            dtype = base.dtype
        head, body = self.link.up.recv()                          # ---- cloud side
        if head[0] == UP_BF16:
            rows = torch.frombuffer(body, dtype=torch.bfloat16).view(1, r, d).to(self.cloud)
            rows = rows.to(dtype)
        else:
            qc = torch.frombuffer(body, dtype=torch.int8, count=r * d).view(1, r, d)
            sc = torch.frombuffer(body, dtype=torch.float16, count=r, offset=r * d).view(1, r, 1)
            qc, sc = qc.to(self.cloud).float(), sc.to(self.cloud).float()
            rows = (base.float() + qc * sc).to(dtype)
        _sync(self.cloud)
        ms = (time.perf_counter() - t0) * 1e3
        return rows, {"up_wire_ms": ms, "wire_up_bytes": sent, "edge_sleep_ms": slept}
