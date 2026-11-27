"""Exp21: DASR as two programs -- an edge program and a cloud program -- over TCP.

Until Exp20 both halves ran in ONE Python process (two GPUs, or one GPU); here they
are separate programs that can run on separate machines (e.g. a rented RTX 3060 as
the edge and the lab server as the cloud).

Roles (one TCP connection per edge device, opened by the edge, which has the prompt):

  edge  (EdgeRunner)   tokenizer, embedding + Dream layers 1..k, its own caches for
                       those layers, the codec; answers the cloud's step requests.
  cloud (CloudSession) Dream layers k+1..L, LM head, sampler, the cloud caches and the
                       refresh / block decisions; drives the generation.

Per denoising step (one round trip):
  cloud -> edge  STEP  the tokens committed since the last step + what to compute:
                       which rows (chunk), cache mode (Fast-dLLM block start / in-block),
                       dLLM-Cache refresh flags, which rows to send, bf16 or INT8.
  edge  -> cloud ROWS  the boundary rows (bf16, or INT8 deltas) + the edge's compute time
At the end the cloud sends DONE with the last commits, so the edge holds the answer.

The cloud runs the UNCHANGED samplers (src/feature_cache_ext.generate for dLLM-Cache,
src/fastdllm_split.kv_generate for Fast-dLLM). Its model keeps only layers k+1..L, and
its embedding is replaced by RemoteEmbedding, which returns the boundary rows received
from the edge. RemoteLink has the same interface as SplitLink / KVBlockLink (begin,
set_chunk, end_step, end), so the samplers drive it exactly as before.

The edge reproduces what layers 1..k did in the single-process runs:
  * dLLM-Cache: the official hook on its k layers, with the refresh decisions of each
    step taken from STEP (the cloud's controller decides, as before);
  * Fast-dLLM: its own key/value cache for its k layers (block start = full forward that
    stores it; in-block = forward over the block with replace_position).
With bf16 rows the answers must equal the single-process runs on the same GPU type.

Wire format (little endian):
  frame  = <BI type, body length> + body
  BEGIN  JSON (prompt ids and the generation settings)            edge  -> cloud
  STEP   <IBBBIIIIIIBI> step, mode, refresh_gen, refresh_prompt, chunk_lo, chunk_hi,
         send_from, send_to, rp_lo, rp_hi, raw, n_commits + positions (uint16) +
         token ids (uint32)                                        cloud -> edge
  ROWS   <fBII3x> edge_ms, raw, rows, d (16 B) + bf16 rows | INT8 deltas + fp16 scales
  DONE   JSON (last commits and the cloud's per-answer statistics)  cloud -> edge
  BYE    empty                                                     edge  -> cloud

Several edges at once (Exp22): one cloud model serves every connection in its own thread
and CUDA stream. ThreadLinks routes the model's embedding and LM-head hook to the calling
thread's RemoteLink (created with hook=False), and all waits are per stream, so one
user's step never waits for another user's kernels.
"""
from __future__ import annotations

import json
import socket
import struct
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np

F_BEGIN, F_STEP, F_ROWS, F_DONE, F_BYE = 1, 2, 3, 4, 5
HDR = struct.Struct("<BI")
STEP = struct.Struct("<IBBBIIIIIIBI")
ROWS = struct.Struct("<fBII3x")    # 16 bytes: keeps the payload 2-byte aligned
MODE_FULL, MODE_FULL_STORE, MODE_BLOCK = 0, 1, 2


# --------------------------------------------------------------------------- framing
def _recv_exact(sock, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise EOFError("connection closed")
        got += k
    return buf


def _as_bytes(p):
    if isinstance(p, np.ndarray):
        p = np.ascontiguousarray(p).reshape(-1).view(np.uint8)
    return memoryview(p).cast("B")


def send_frame(sock, ftype, *parts):
    """Send one frame; returns its size in bytes (header included)."""
    views = [_as_bytes(p) for p in parts]
    n = sum(v.nbytes for v in views)
    sock.sendall(HDR.pack(ftype, n))
    for v in views:
        if v.nbytes:
            sock.sendall(v)
    return HDR.size + n


def recv_frame(sock):
    ftype, n = HDR.unpack(_recv_exact(sock, HDR.size))
    return ftype, (_recv_exact(sock, n) if n else bytearray()), HDR.size + n


def connect(addr, wait_s=900):
    """Connect to a cloud program, retrying while it is still loading (up to wait_s)."""
    host, port = addr.rsplit(":", 1)
    deadline = time.time() + wait_s
    while True:
        try:
            s = socket.create_connection((host, int(port)), timeout=30)
            break
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(5)
    s.settimeout(None)
    _tune(s)
    return s


def _tune(s):
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        s.setsockopt(socket.SOL_SOCKET, opt, 4 << 20)


def pack_step(f, pos, tok):
    pos = np.asarray(pos, dtype="<u2")
    tok = np.asarray(tok, dtype="<u4")
    head = STEP.pack(f["step"], f["mode"], f["g"], f["p"], f["chunk_lo"], f["chunk_hi"],
                     f["send_from"], f["send_to"], f["rp_lo"], f["rp_hi"], f["raw"], len(pos))
    return head, pos, tok


def unpack_step(body):
    v = STEP.unpack_from(body, 0)
    keys = ("step", "mode", "g", "p", "chunk_lo", "chunk_hi", "send_from", "send_to",
            "rp_lo", "rp_hi", "raw", "n")
    f = dict(zip(keys, v))
    n, off = f["n"], STEP.size
    pos = np.frombuffer(body, dtype="<u2", count=n, offset=off).astype(np.int64)
    tok = np.frombuffer(body, dtype="<u4", count=n, offset=off + 2 * n).astype(np.int64)
    return f, pos, tok


# --------------------------------------------------------------------------- shared
def import_dllm_cache(repo):
    """(dLLMCache, dLLMCacheConfig, register_cache_Dream, logout_cache_Dream) from the
    official dLLM-Cache checkout at `repo` (as exp9b.import_official)."""
    repo = str(Path(repo).expanduser().resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from dllm_cache.cache import dLLMCache, dLLMCacheConfig
    from dllm_cache.hooks import logout_cache_Dream, register_cache_Dream
    return dLLMCache, dLLMCacheConfig, register_cache_Dream, logout_cache_Dream


def _sync(dev):
    """Wait for this thread's CUDA stream (with one thread, the same as a device sync)."""
    import torch
    if torch.device(dev).type == "cuda":
        torch.cuda.current_stream(dev).synchronize()


# --------------------------------------------------------------------------- cloud
class RemoteEmbedding:
    """Factory for the cloud model's embedding replacement (an nn.Module)."""

    @staticmethod
    def make(link):
        import torch.nn as nn

        class _RemoteEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self._link = [link]          # list: not registered as a submodule

            def forward(self, input_ids):
                return self._link[0].remote_step(input_ids)

        return _RemoteEmbedding()


class ThreadLinks:
    """For a cloud program that serves several edges at once (Exp22): every serving
    thread sets its own RemoteLink (made with hook=False); the model's embedding
    (strip_edge(model, k, router)) and its LM-head hook use the calling thread's link."""

    def __init__(self, model):
        self._local = threading.local()
        model.lm_head.register_forward_hook(self._on_head)

    def set(self, link):
        self._local.link = link

    def remote_step(self, input_ids):
        return self._local.link.remote_step(input_ids)

    def _on_head(self, module, args, output):
        link = getattr(self._local, "link", None)
        if link is not None:
            link._on_head(module, args, output)
        return None


def strip_edge(model, k, link):
    """Cloud model: drop the embedding and layers 1..k, read the boundary from the edge."""
    import torch
    import torch.nn as nn
    base = model.model
    keep = list(base.layers)[k:]
    base.layers = nn.ModuleList(keep)
    base.embed_tokens = RemoteEmbedding.make(link)
    torch.cuda.empty_cache()
    return model


class RemoteLink:
    """Cloud side of the connection, with the interface of SplitLink / KVBlockLink.

    sock     the edge's connection (set per session)
    The dLLM-Cache sampler calls begin(prompt_len=..., controller=..., rows=..., encoding=...,
    prompt_tail_rows=..., mask_id=..., anchor_interval=...); kv_generate calls
    begin(n_total=..., n_prompt=..., encoding=..., anchor_interval=..., cache=...) and
    set_chunk(offset, kind, block, span) before every model call.
    """

    def __init__(self, model, device, d, hook=True):
        self.device, self.d = device, int(d)
        self.edge = device          # kv_generate builds replace_position on link.edge
        self.sock = None
        self.active = False
        self.cloud_end = None
        if hook:                    # hook=False: ThreadLinks calls _on_head instead
            model.lm_head.register_forward_hook(self._on_head)

    # -------------------------------------------------------------- session
    def begin(self, **kw):
        import torch
        self.kv = "n_total" in kw
        self.encoding = kw.get("encoding", "bf16")
        if self.encoding not in {"bf16", "int8_delta"}:
            raise ValueError(f"remote link supports bf16 and int8_delta, not {self.encoding}")
        self.anchor_interval = int(kw.get("anchor_interval", 32))
        if self.kv:
            self.n_total = int(kw["n_total"])
            self.cache = kw.get("cache", "dual")
        else:
            self.rows = kw.get("rows", "all")
            self.controller = kw.get("controller")
            self.split = int(kw["prompt_len"]) - int(kw.get("prompt_tail_rows", 0))
            if self.rows == "cache_aware" and self.controller is None:
                raise ValueError("rows='cache_aware' needs the feature cache")
            self.n_total = None
        self.mirror = None            # (1, n, d): the boundary rows the cloud holds
        self.record = None            # (1, n): the tokens the edge holds
        self.step, self.last_anchor = 0, 0
        self.offset, self.kind = 0, "full"
        self.stats, self.ev = [], []
        self.active = True
        self.torch = torch

    def set_chunk(self, offset, kind, block, span=None):
        self.offset, self.kind = int(offset), kind

    def end_step(self, n_commits):
        if self.active and self.stats:
            self.stats[-1]["n_commits"] = int(n_commits)

    def end(self):
        if not self.active:
            return []
        torch = self.torch
        tail = torch.cuda.Event(enable_timing=True)
        tail.record(torch.cuda.current_stream(self.device))
        tail.synchronize()
        out = []
        for t, (ev, st) in enumerate(zip(self.ev, self.stats)):
            nxt = self.ev[t + 1]["sample_end"] if t + 1 < len(self.ev) else tail
            row = dict(st)
            row["cloud_ms"] = ev["cloud_start"].elapsed_time(ev["cloud_end"]) if "cloud_end" in ev else 0.0
            row["sample_ms"] = ev["cloud_end"].elapsed_time(nxt) if "cloud_end" in ev else 0.0
            row.setdefault("n_commits", 0)
            out.append(row)
        self.active = False
        return out

    def _on_head(self, module, args, output):
        if self.active and self.ev:
            e = self.torch.cuda.Event(enable_timing=True)
            e.record(self.torch.cuda.current_stream(self.device))
            self.ev[-1]["cloud_end"] = e
        return None

    def _event(self):
        e = self.torch.cuda.Event(enable_timing=True)
        e.record(self.torch.cuda.current_stream(self.device))
        return e

    def _cache_step(self):
        """dLLM-Cache step about to run (counters of the cloud layers + 1)."""
        from dllm_cache.cache import dLLMCache
        inst = dLLMCache()
        cnt = inst._dLLMCache__step_counter[inst.cache_type]
        return (max(cnt.values()) if cnt else 0) + 1

    # -------------------------------------------------------------- one step
    def remote_step(self, input_ids):
        torch = self.torch
        if not self.active:
            raise RuntimeError("RemoteLink used outside a session")
        ev = {"sample_end": self._event()}
        self.step += 1
        ids = input_ids
        m = int(ids.shape[1])
        if self.kv:
            lo = self.offset
            mode = (MODE_BLOCK if self.kind == "block" else
                    (MODE_FULL_STORE if self.cache == "dual" else MODE_FULL))
            send_from, send_to = lo, lo + m
            g = p = 0
            raw = (self.step == 1 or self.encoding == "bf16"
                   or self.step - self.last_anchor >= self.anchor_interval)
            rp = (lo, lo + m) if mode == MODE_BLOCK else (0, 0)
            n = self.n_total
        else:
            lo, mode, n = 0, MODE_FULL, m
            first = self.mirror is None
            if self.controller is not None:
                g, p = self.controller.decide(self._cache_step())
                g, p = int(bool(g)), int(bool(p))
            else:
                g = p = 1
            prompt_now = self.rows == "all" or first or bool(p)
            send_from, send_to = (0 if prompt_now else self.split), n
            raw = (first or (self.rows == "cache_aware" and prompt_now) or self.encoding == "bf16"
                   or self.step - self.last_anchor >= self.anchor_interval)
            rp = (0, 0)
        if raw and self.encoding != "bf16":
            self.last_anchor = self.step
        # tokens committed since the last step (the edge's copy of the sequence)
        _sync(self.device)
        t0 = time.perf_counter()
        if self.record is None:  # first step: the edge built the same canvas itself
            if lo != 0 or m != n:
                raise RuntimeError("the first step must cover the whole sequence")
            self.record = ids.detach().clone()
        cur = self.record[:, lo:lo + m]
        idx = torch.nonzero(ids[0] != cur[0]).squeeze(1)
        vals = ids[0, idx]
        self.record[:, lo:lo + m] = ids
        f = {"step": self.step, "mode": mode, "g": g, "p": p, "chunk_lo": lo, "chunk_hi": lo + m,
             "send_from": send_from, "send_to": send_to, "rp_lo": rp[0], "rp_hi": rp[1],
             "raw": int(bool(raw))}
        down = send_frame(self.sock, F_STEP, *pack_step(f, (idx + lo).cpu().numpy(),
                                                        vals.cpu().numpy()))
        ftype, body, up = recv_frame(self.sock)
        if ftype != F_ROWS:
            raise RuntimeError(f"expected ROWS, got frame type {ftype}")
        edge_ms, raw_r, r, d = ROWS.unpack_from(body, 0)
        rows = self._decode(body, ROWS.size, bool(raw_r), r, d, send_from, send_to)
        if self.mirror is None:
            if send_from != 0 or send_to != n:
                raise RuntimeError("the first step must send every row")
            self.mirror = rows.clone()
        else:
            self.mirror[:, send_from:send_to] = rows
        hidden = self.mirror[:, lo:lo + m].clone()
        _sync(self.device)
        wire_ms = (time.perf_counter() - t0) * 1e3
        ev["cloud_start"] = self._event()
        self.ev.append(ev)
        self.stats.append({"step": self.step, "kind": self.kind if self.kv else "full",
                           "rows_sent": int(r), "raw": bool(raw), "refresh_gen": g,
                           "refresh_prompt": p, "up_bytes": int(up), "down_bytes": int(down),
                           "edge_ms": float(edge_ms), "wire_ms": wire_ms})
        return hidden

    def _decode(self, body, off, raw, r, d, lo, hi):
        torch = self.torch
        if raw:
            rows = torch.frombuffer(body, dtype=torch.bfloat16, count=r * d, offset=off)
            return rows.view(1, r, d).to(self.device)
        q = torch.frombuffer(body, dtype=torch.int8, count=r * d, offset=off).view(1, r, d)
        s = torch.frombuffer(body, dtype=torch.float16, count=r, offset=off + r * d).view(1, r, 1)
        base = self.mirror[:, lo:hi]
        return (base.float() + q.to(self.device).float() * s.to(self.device).float()).to(base.dtype)


# --------------------------------------------------------------------------- edge
@contextmanager
def edge_cache_session(model, api, prompt_len, prompt_tail_rows, transfer_ratio, flags):
    """Official dLLM-Cache hook on the edge's k layers; refresh decisions = flags[0:2],
    which the edge sets from every STEP (the cloud's controller decides)."""
    import torch
    dLLMCache, dLLMCacheConfig, register, logout = api
    split = int(prompt_len) - int(prompt_tail_rows)
    inst = dLLMCache.new_instance(**asdict(dLLMCacheConfig(
        prompt_interval_steps=1, gen_interval_steps=1, transfer_ratio=float(transfer_ratio))))
    for attr in ("refresh_gen", "refresh_prompt"):
        inst.__dict__.pop(attr, None)
    inst.refresh_gen = lambda layer_id=0: bool(flags[0])
    inst.refresh_prompt = lambda layer_id=0: bool(flags[1])
    hook_mod = sys.modules["dllm_cache.hooks.cache_hook_Dream"]
    original = hook_mod.refresh_index
    if prompt_tail_rows > 0:   # as feature_cache_session(force_prompt_tail=True)
        def refresh_index(new_features, cached_features=None, transfer_ratio=0.5, layer_id=0):
            _, gen_len, _ = new_features.shape
            kk = int(gen_len * transfer_ratio)
            cos = torch.nn.functional.cosine_similarity(new_features, cached_features, dim=-1)
            cos[:, :prompt_tail_rows] = -2.0
            return torch.topk(cos, largest=False, k=kk).indices
        hook_mod.refresh_index = refresh_index
    registered = False
    try:
        register(model, "model.layers")
        registered = True
        inst.reset_cache(split)
        yield inst
    finally:
        if registered:
            logout(model, "model.layers")
        hook_mod.refresh_index = original
        for attr in ("refresh_gen", "refresh_prompt"):
            inst.__dict__.pop(attr, None)


def load_edge_model(backend, model_name, k, device, fastdllm_repo=None):
    """Embedding + Dream's first k layers only (weights read from the checkpoint shards
    that hold them); final norm and LM head replaced by identities, so the model's
    output 'logits' are the boundary rows h_k."""
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    if backend == "fastdllm":
        from src.fastdllm_split import import_fastdllm
        pkg = import_fastdllm("dream", fastdllm_repo)
        cfg = pkg.DreamConfig.from_pretrained(model_name)
        build = lambda c: pkg.DreamModel(c)  # noqa: E731
    else:
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        build = lambda c: AutoModel.from_config(c, trust_remote_code=True)  # noqa: E731
    cfg.num_hidden_layers = int(k)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            model = build(cfg)
    finally:
        torch.set_default_dtype(prev)
    model.lm_head = nn.Identity()
    model.model.norm = nn.Identity()
    index = json.loads(Path(hf_hub_download(model_name, "model.safetensors.index.json"))
                       .read_text())["weight_map"]
    wanted = [key for key in index if key.startswith("model.embed_tokens.")
              or any(key.startswith(f"model.layers.{i}.") for i in range(int(k)))]
    tensors = {}
    for shard in sorted({index[key] for key in wanted}):
        path = hf_hub_download(model_name, shard)
        with safe_open(path, framework="pt", device=str(device)) as f:
            for key in wanted:
                if index[key] == shard:
                    tensors[key] = f.get_tensor(key).to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    bad = [m for m in missing if m.startswith(("model.embed_tokens", "model.layers"))]
    if bad or unexpected:
        raise RuntimeError(f"edge weights incomplete: missing {bad[:5]}, unexpected {unexpected[:5]}")
    torch.cuda.empty_cache()
    return model.eval()


class EdgeRunner:
    """Edge side of one connection: runs sessions (one per answer) on request."""

    def __init__(self, models, device, api=None, slowdown=1.0):
        """models: {"dllm_cache": edge model (HF Dream), "fastdllm": edge model (Fast-dLLM)}"""
        import torch
        self.torch = torch
        self.models, self.device, self.api = models, torch.device(device), api
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.slow = float(slowdown)

    def session(self, sock, cfg, prompt_ids, mask_id):
        """Send BEGIN, answer STEP requests until DONE. Returns (answer ids, info)."""
        torch = self.torch
        backend = cfg["backend"]
        plen = len(prompt_ids)
        n = plen + int(cfg["new_tokens"])
        canvas = torch.full((1, n), int(mask_id), dtype=torch.long)
        canvas[0, :plen] = torch.tensor(prompt_ids, dtype=torch.long)
        body = json.dumps({**cfg, "prompt_ids": list(map(int, prompt_ids))}).encode()
        t0 = time.perf_counter()
        sent = send_frame(sock, F_BEGIN, body)
        model = self.models.get(backend) if cfg.get("split", 0) > 0 else None
        flags = [1, 1]
        state = {"ref": None, "pkv": None, "edge_ms": 0.0, "steps": 0, "up": 0,
                 "down": 0, "sleep_ms": 0.0}
        ctx = (edge_cache_session(model, self.api, plen, int(cfg.get("prompt_tail_rows", 0)),
                                  float(cfg.get("transfer_ratio", 0.25)), flags)
               if (model is not None and backend == "dllm_cache" and cfg.get("policy"))
               else _null())
        with ctx, torch.inference_mode():
            while True:
                ftype, body, size = recv_frame(sock)
                state["down"] += size
                if ftype == F_DONE:
                    info = json.loads(bytes(body).decode())
                    pos, tok = info.pop("final_pos"), info.pop("final_tok")
                    canvas[0, pos] = torch.tensor(tok, dtype=torch.long)
                    break
                if ftype != F_STEP:
                    raise RuntimeError(f"edge: unexpected frame type {ftype}")
                f, pos, tok = unpack_step(body)
                canvas[0, torch.from_numpy(pos)] = torch.from_numpy(tok)
                flags[0], flags[1] = f["g"], f["p"]
                state["up"] += self._step(sock, model, backend, canvas, f, state)
                state["steps"] += 1
        wall = time.perf_counter() - t0
        info.update({"edge_wall_s": wall, "edge_steps": state["steps"],
                     "edge_compute_ms": state["edge_ms"], "edge_sleep_ms": state["sleep_ms"],
                     "edge_up_bytes": state["up"] + sent, "edge_down_bytes": state["down"]})
        return canvas[0, plen:].clone(), info

    def _step(self, sock, model, backend, canvas, f, state):
        torch = self.torch
        dev = self.device
        lo, hi = f["chunk_lo"], f["chunk_hi"]
        ids = canvas[:, lo:hi].to(dev)
        _sync(dev)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        if backend == "fastdllm":
            if f["mode"] == MODE_BLOCK:
                rp = torch.zeros((1, canvas.shape[1]), dtype=torch.bool, device=dev)
                rp[:, f["rp_lo"]:f["rp_hi"]] = True
                out = model(ids, "full", None, past_key_values=state["pkv"], use_cache=True,
                            dual_cache=True, replace_position=rp)
            else:
                store = f["mode"] == MODE_FULL_STORE
                out = model(ids, "full", None, use_cache=store)
                if store:
                    state["pkv"] = out.past_key_values
        else:
            out = model(ids, "full", None)
        h = out.logits.to(torch.bfloat16)
        rows = h[:, f["send_from"] - lo:f["send_to"] - lo]
        r, d = int(rows.shape[1]), int(rows.shape[2])
        a, b = f["send_from"], f["send_to"]
        if state["ref"] is None:
            state["ref"] = torch.zeros(1, canvas.shape[1], d, dtype=torch.bfloat16, device=dev)
        if f["raw"]:
            state["ref"][:, a:b] = rows
            payload = [rows.contiguous().view(torch.int16).cpu().numpy()]
        else:
            ref = state["ref"][:, a:b].float()
            delta = rows.float() - ref
            scale = delta.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
            q = torch.clamp(torch.round(delta / scale), -127, 127)
            s16 = scale.half()
            state["ref"][:, a:b] = (ref + q * s16.float()).to(torch.bfloat16)
            payload = [q.to(torch.int8).contiguous().cpu().numpy(),
                       s16.reshape(-1).contiguous().cpu().numpy()]
        e1.record()
        _sync(dev)
        ms = e0.elapsed_time(e1)
        state["edge_ms"] += ms
        if self.slow > 1.0:
            sl = (self.slow - 1.0) * ms
            time.sleep(sl / 1e3)
            state["sleep_ms"] += sl
        return send_frame(sock, F_ROWS, ROWS.pack(float(ms), int(f["raw"]), r, d), *payload)


@contextmanager
def _null():
    yield None
