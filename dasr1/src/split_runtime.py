"""DASR two-GPU split runtime (simulated edge = one GPU, cloud = another GPU).

Closed loop: Dream's own ``diffusion_generate`` runs unchanged; module hooks place
and connect the two halves:

  edge  (EDGE GPU):  embed_tokens, rotary_emb, layers[0 : k]       (+ dLLM-Cache hook)
  cloud (CLOUD GPU): layers[k : L], norm, lm_head, sampler        (+ dLLM-Cache hook)

Per denoising step:
  downlink  cloud -> edge : the tokens committed in the previous step
  uplink    edge -> cloud : the boundary hidden state H_k, encoded by a codec

Boundary codec = (rows, encoding):
  rows="all"          every row, every step (static split baseline)
  rows="cache_aware"  answer rows (+ prompt-tail rows) every step; PROMPT rows only
                      on prompt-refresh steps of the feature cache. Between those
                      steps the cloud layers read prompt K/V and attention/MLP
                      outputs from their own cache, and the stale prompt rows only
                      travel along the residual path, so gen-row logits are
                      unchanged. Requires the feature cache and prompt_tail_rows>=1
                      (Dream predicts answer token 0 from the last prompt row).
                      Expected EXACT w.r.t. rows="all" with the same cache policy;
                      Exp11 verifies this.
  encoding="bf16"         native rows
  encoding="exact_delta"  only rows that differ bitwise from the last synchronized
                          boundary (+4-byte index each); exact
  encoding="int8_delta"   per-row INT8 delta vs the reconstructed reference with an
                          fp16 scale (closed loop, so quantization error does not
                          accumulate); bf16 anchors at step 1, at prompt refreshes and
                          every `anchor_interval` steps; APPROXIMATE

Simulation notes (state these in the paper):
  * Bytes are counted from the wire format above (+16-byte header per uplink packet,
    6 bytes per committed token + 8-byte header per downlink packet). The tensors
    actually copied between the GPUs are the reconstructed values; copy time over
    PCIe/NVLink is NOT the link model. Network latency is added afterwards by
    experiments/exp11b_network_emulation.py from the recorded bytes.
  * Compute time per side is measured with CUDA events on each GPU: edge_ms (embed
    -> layer k-1), encode_ms (codec on the edge), cloud_ms (layer k -> lm_head),
    sample_ms (sampler on the cloud, between lm_head and the next step).
"""
from __future__ import annotations

import torch

UP_HEADER_B = 16
DOWN_HEADER_B = 8
DOWN_PER_TOKEN_B = 6  # uint16 position + uint32 token id


def _move(obj, device, memo):
    if isinstance(obj, torch.Tensor):
        if obj.device == device:
            return obj
        key = id(obj)
        if key not in memo:
            memo[key] = (obj, obj.to(device))  # keep obj alive so id() is not reused
        return memo[key][1]
    if isinstance(obj, tuple):
        return tuple(_move(o, device, memo) for o in obj)
    if isinstance(obj, list):
        return [_move(o, device, memo) for o in obj]
    if isinstance(obj, dict):
        return {k: _move(v, device, memo) for k, v in obj.items()}
    return obj


def place_split(model, split_layer, edge_device, cloud_device):
    """Move the two halves to their GPUs and make model.device report the cloud.

    split_layer k uses the README's 1-based numbering: the edge runs layers 1..k.
    """
    edge, cloud = torch.device(edge_device), torch.device(cloud_device)
    base = model.model
    n_layers = len(base.layers)
    if not 1 <= split_layer < n_layers:
        raise ValueError(f"split_layer must be in [1, {n_layers - 1}]")
    if model.lm_head.weight.data_ptr() == base.embed_tokens.weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    base.embed_tokens.to(edge)
    base.rotary_emb.to(edge)
    for i, layer in enumerate(base.layers):
        layer.to(edge if i < split_layer else cloud)
    base.norm.to(cloud)
    model.lm_head.to(cloud)
    # Dream's sampler allocates its work tensors on `self.device` (first parameter =
    # embedding = edge). The sampler belongs to the cloud, so report the cloud.
    cls = model.__class__
    if not getattr(cls, "_dasr_split", False):
        model.__class__ = type(cls.__name__, (cls,),
                               {"device": property(lambda self: self._dasr_device),
                                "_dasr_split": True})
    model._dasr_device = cloud
    torch.cuda.empty_cache()
    return edge, cloud


class SplitLink:
    """Hooks that connect the edge and cloud halves and log every step."""

    def __init__(self, model, split_layer, edge, cloud):
        self.model, self.k, self.edge, self.cloud = model, int(split_layer), edge, cloud
        base = model.model
        self.d = int(base.embed_tokens.weight.shape[1])
        self.handles = [
            base.embed_tokens.register_forward_pre_hook(self._on_step_start),
            base.layers[self.k].register_forward_pre_hook(self._on_boundary, with_kwargs=True),
            model.lm_head.register_forward_hook(self._on_cloud_end),
        ]
        for layer in base.layers[self.k + 1:]:
            self.handles.append(layer.register_forward_pre_hook(self._on_cloud_layer,
                                                                with_kwargs=True))
        self.active = False

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    # ------------------------------------------------------------------ setup
    def begin(self, *, prompt_len, rows="all", encoding="bf16", controller=None,
              prompt_tail_rows=0, mask_id=None, anchor_interval=32):
        if rows not in {"all", "cache_aware"}:
            raise ValueError(f"rows={rows!r}")
        if encoding not in {"bf16", "exact_delta", "int8_delta"}:
            raise ValueError(f"encoding={encoding!r}")
        if rows == "cache_aware":
            if controller is None:
                raise ValueError("rows='cache_aware' needs the feature cache (a policy)")
            if prompt_tail_rows < 1:
                raise ValueError("rows='cache_aware' needs prompt_tail_rows >= 1 on Dream "
                                 "(answer token 0 is predicted from the last prompt row)")
        self.rows, self.encoding, self.controller = rows, encoding, controller
        self.split = int(prompt_len) - int(prompt_tail_rows)  # first row sent every step
        self.mask_id = mask_id
        self.anchor_interval = int(anchor_interval)
        self.ref_edge = None      # edge copy of the synchronized boundary
        self.buf_cloud = None     # cloud copy (what the cloud layers actually read)
        self.prev_ids = None
        self.last_anchor = 0
        self.step = 0
        self.memo = {}
        self.ev = []              # per step dict of CUDA events
        self.stats = []           # per step wire statistics
        self.active = True

    def end(self):
        if not self.active:
            return []
        tail = torch.cuda.Event(enable_timing=True)
        tail.record(torch.cuda.current_stream(self.cloud))
        torch.cuda.synchronize(self.edge)
        torch.cuda.synchronize(self.cloud)
        rows = []
        for t, (ev, st) in enumerate(zip(self.ev, self.stats)):
            nxt = self.ev[t + 1]["sample_end"] if t + 1 < len(self.ev) else tail
            row = dict(st)
            row["edge_ms"] = ev["edge_start"].elapsed_time(ev["edge_end"])
            row["encode_ms"] = ev["edge_end"].elapsed_time(ev["encode_end"])
            row["cloud_ms"] = ev["cloud_start"].elapsed_time(ev["cloud_end"])
            row["sample_ms"] = ev["cloud_end"].elapsed_time(nxt)
            row["n_commits"] = int(st["n_commits"]) if isinstance(st["n_commits"], int) \
                else int(st["n_commits"].item())
            row["down_bytes"] = (DOWN_HEADER_B + DOWN_PER_TOKEN_B * row["n_commits"]
                                 if t > 0 else 0)  # step 1: prompt upload, not counted
            rows.append(row)
        self.active = False
        return rows

    # ------------------------------------------------------------------ hooks
    def _event(self, device):
        e = torch.cuda.Event(enable_timing=True)
        e.record(torch.cuda.current_stream(device))
        return e

    def _on_step_start(self, module, args):
        if not self.active:
            return None
        ids = args[0]
        ev = {"sample_end": self._event(self.cloud)}
        if self.prev_ids is None or self.mask_id is None:
            n_commits = 0
        else:
            n_commits = ((self.prev_ids == self.mask_id) & (ids != self.mask_id)).sum()
        self.prev_ids = ids.detach().clone()
        self.step += 1
        self.memo = {}
        self.stats.append({"step": self.step, "n_commits": n_commits})
        ids_edge = ids.to(self.edge)
        ev["edge_start"] = self._event(self.edge)
        self.ev.append(ev)
        return (ids_edge,) + tuple(args[1:])

    def _cache_step(self):
        """dLLMCache's step counter (set by feature_cache_session); our own as fallback."""
        step_fn = getattr(self.controller, "step_fn", None)
        return step_fn() if step_fn is not None else self.step

    def _prompt_refresh_now(self):
        if self.controller is None:
            return True
        return bool(self.controller.decide(self._cache_step())[1])

    def _on_boundary(self, module, args, kwargs):
        if not self.active:
            return None
        ev = self.ev[-1]
        ev["edge_end"] = self._event(self.edge)
        h = args[0]
        n, d = h.shape[1], h.shape[2]
        row_b = d * h.element_size()
        first = self.ref_edge is None
        prompt_now = self.rows == "all" or first or self._prompt_refresh_now()
        lo = 0 if prompt_now else self.split          # rows to send: [lo, n)
        anchor = first or (self.rows == "cache_aware" and prompt_now) or (
            self.encoding == "int8_delta"
            and self.step - self.last_anchor >= self.anchor_interval)
        if first:
            self.ref_edge = h.detach().clone()
        sel = h[:, lo:]
        if anchor or self.encoding == "bf16":
            n_rows, payload = n - lo, (n - lo) * row_b
            new = sel
        elif self.encoding == "exact_delta":
            changed = (sel != self.ref_edge[:, lo:]).any(dim=-1)
            n_rows_t = changed.sum()
            n_rows = n_rows_t  # tensor; resolved below
            payload = None
            new = sel
        else:  # int8_delta
            delta = sel.float() - self.ref_edge[:, lo:].float()
            scale = delta.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
            q = torch.clamp(torch.round(delta / scale), -127, 127)
            scale16 = scale.half().float()
            new = (self.ref_edge[:, lo:].float() + q * scale16).to(h.dtype)
            n_rows, payload = n - lo, (n - lo) * (d + 2)
        if self.encoding != "bf16":  # delta codecs need the synchronized reference
            self.ref_edge[:, lo:] = new
        if anchor:
            self.last_anchor = self.step
        ev["encode_end"] = self._event(self.edge)

        # "transmit": the cloud keeps its own buffer; rows outside [lo, n) stay stale
        if self.buf_cloud is None:
            self.buf_cloud = self.ref_edge.to(self.cloud)
        else:
            self.buf_cloud[:, lo:] = new.to(self.cloud)
        if self.encoding == "exact_delta" and payload is None:
            n_rows = int(n_rows.item())
            payload = n_rows * (row_b + 4)
        self.stats[-1].update({
            "rows_sent": int(n_rows), "prompt_sent": bool(lo == 0), "anchor": bool(anchor),
            "up_bytes": UP_HEADER_B + int(payload), "full_bytes": n * row_b,
        })
        if self.controller is not None:
            g, p = self.controller.decide(self._cache_step())
            self.stats[-1].update({"gen_refresh": bool(g), "prompt_refresh": bool(p)})
        hidden = self.buf_cloud.clone()
        kwargs = _move(kwargs, self.cloud, self.memo)
        rest = _move(tuple(args[1:]), self.cloud, self.memo)
        ev["cloud_start"] = self._event(self.cloud)
        return (hidden,) + rest, kwargs

    def _on_cloud_layer(self, module, args, kwargs):
        if not self.active:
            return None
        return _move(tuple(args), self.cloud, self.memo), _move(kwargs, self.cloud, self.memo)

    def _on_cloud_end(self, module, args, output):
        if self.active:
            self.ev[-1]["cloud_end"] = self._event(self.cloud)
        return None
