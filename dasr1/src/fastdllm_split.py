"""Fast-dLLM (block-wise KV cache + parallel decoding) on the DASR two-GPU split (Exp13).

Fast-dLLM (NVlabs, v1) changes two things that matter for a split:

  * DualCache: at the start of every block of `block_length` answer tokens the
    model runs once over the whole sequence and keeps the keys/values of every
    layer; inside the block it only runs the block's rows, attending to the
    cached prefix and suffix. With the row rule of the paper, S_t = R_t u L_t,
    the boundary of an in-block step is exactly the block's rows: the edge sends
    `block_length` rows instead of n.
  * Parallel decoding: inside a block, every masked position whose confidence is
    >= threshold gamma is committed at once (at least one per step), so a block
    needs far fewer denoising steps -> fewer network round trips.

Both halves keep the KV cache of their own layers on their own GPU; only the
boundary rows of the rows being computed cross the link.

This module provides
  load_fastdllm_model(arch, name, repo)   Fast-dLLM's modified Dream / LLaDA model
  KVBlockLink                              split link for variable-length forwards
  kv_generate(...)                         DualCache / no-cache x fixed / threshold
                                           decoding, Dream logit shift handled as in
                                           Fast-dLLM (first block token committed at
                                           the block-start full forward)
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.split_runtime import DOWN_HEADER_B, DOWN_PER_TOKEN_B, UP_HEADER_B, _move, model_parts
from src.utils import set_seed

DREAM_MODEL_NAME = "Dream-org/Dream-v0-Instruct-7B"
LLADA_MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
LLADA_MASK_ID = 126336


# --------------------------------------------------------------------------- loading
def import_fastdllm(arch, repo):
    """Import Fast-dLLM's v1/<arch>/model package under a unique name."""
    root = Path(repo).expanduser().resolve() / "v1" / arch / "model"
    if not (root / "__init__.py").is_file():
        raise FileNotFoundError(
            f"Fast-dLLM checkout not found at {root}. Clone it with\n"
            f"  git clone https://github.com/NVlabs/Fast-dLLM.git <path>\n"
            f"and set FASTDLLM_REPO in configs/experiment_config.py to <path>.")
    name = f"fastdllm_{arch}_model"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / "__init__.py",
                                                      submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def load_fastdllm_model(arch, model_name, repo, device="cuda"):
    from transformers import AutoConfig, AutoTokenizer
    pkg = import_fastdllm(arch, repo)
    if arch == "dream":
        model = pkg.DreamModel.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                               trust_remote_code=True)
    elif arch == "llada":
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = pkg.LLaDAModelLM.from_pretrained(model_name, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16, config=config)
    else:
        raise ValueError(f"arch must be 'dream' or 'llada', not {arch!r}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model.to(device).eval(), tokenizer


def mask_token_id(arch, tokenizer):
    if arch == "llada":
        return LLADA_MASK_ID
    if tokenizer.mask_token_id is None:
        raise RuntimeError("Dream tokenizer has no mask_token_id")
    return int(tokenizer.mask_token_id)


# --------------------------------------------------------------------------- link
class KVBlockLink:
    """Edge-cloud link for forwards over a variable chunk of rows.

    The generator calls set_chunk(offset, kind, block) before every model call; the
    boundary hook then transmits all rows of that chunk (exactly the rows the cloud
    computes and reads). Encodings:
      bf16        native rows                                         (exact)
      int8_delta  per-row INT8 delta vs the reference both sides hold at the same
                  absolute positions, fp16 scale, closed loop; bf16 anchor at the
                  first step and at least every `anchor_interval` steps (approximate)
    Per-step statistics use the same columns as src.split_runtime.SplitLink, so
    experiments/exp11b_network_emulation.py works unchanged.
    """

    def __init__(self, model, split_layer, edge, cloud):
        self.k, self.edge, self.cloud = int(split_layer), edge, cloud
        parts = model_parts(model)
        self.handles = [
            parts.embed.register_forward_pre_hook(self._on_step_start),
            parts.layers[self.k].register_forward_pre_hook(self._on_boundary, with_kwargs=True),
            parts.head.register_forward_hook(self._on_cloud_end),
        ]
        for layer in parts.layers[self.k + 1:]:
            self.handles.append(layer.register_forward_pre_hook(self._on_cloud_layer,
                                                                with_kwargs=True))
        self.active = False

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def begin(self, *, n_total, encoding="bf16", anchor_interval=32):
        if encoding not in {"bf16", "int8_delta"}:
            raise ValueError(f"encoding={encoding!r}")
        self.n_total, self.encoding = int(n_total), encoding
        self.anchor_interval = int(anchor_interval)
        self.ref = None          # (1, n_total, d) on the edge; the cloud holds the same values
        self.last_anchor = 0
        self.step = 0
        self.offset, self.kind, self.block = 0, "full", 0
        self.memo, self.ev, self.stats = {}, [], []
        self.active = True

    def set_chunk(self, offset, kind, block):
        self.offset, self.kind, self.block = int(offset), kind, int(block)

    def end_step(self, n_commits):
        if self.active and self.stats:
            self.stats[-1]["n_commits"] = int(n_commits)

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
            # committed tokens of this step go back to the edge (positions + ids)
            row["down_bytes"] = DOWN_HEADER_B + DOWN_PER_TOKEN_B * int(row.get("n_commits", 0))
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
        ev = {"sample_end": self._event(self.cloud)}
        self.step += 1
        self.memo = {}
        self.stats.append({"step": self.step, "kind": self.kind, "block": self.block,
                           "offset": self.offset, "n_commits": 0})
        ids_edge = args[0].to(self.edge)
        ev["edge_start"] = self._event(self.edge)
        self.ev.append(ev)
        return (ids_edge,) + tuple(args[1:])

    def _on_boundary(self, module, args, kwargs):
        if not self.active:
            return None
        ev = self.ev[-1]
        ev["edge_end"] = self._event(self.edge)
        h = args[0]
        m, d = h.shape[1], h.shape[2]
        lo, hi = self.offset, self.offset + m
        row_b = d * h.element_size()
        if self.ref is None:
            self.ref = torch.zeros(1, self.n_total, d, dtype=h.dtype, device=h.device)
        anchor = (self.step == 1 or self.encoding == "bf16"
                  or self.step - self.last_anchor >= self.anchor_interval)
        if anchor:
            new = h
            payload = m * row_b
            if self.encoding != "bf16":
                self.last_anchor = self.step
        else:  # int8_delta against the synchronized reference at the same positions
            ref = self.ref[:, lo:hi].float()
            delta = h.float() - ref
            scale = delta.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
            q = torch.clamp(torch.round(delta / scale), -127, 127)
            new = (ref + q * scale.half().float()).to(h.dtype)
            payload = m * (d + 2)
        if self.encoding != "bf16":
            self.ref[:, lo:hi] = new
        ev["encode_end"] = self._event(self.edge)
        self.stats[-1].update({
            "rows_sent": int(m), "anchor": bool(anchor),
            "up_bytes": UP_HEADER_B + int(payload),
            "full_bytes": self.n_total * row_b,   # static split without cache, for reference
        })
        hidden = new.to(self.cloud)
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


# --------------------------------------------------------------------------- sampling
def _top_p_logits(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_indices, remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def propose(arch, logits, temperature, top_p):
    """(confidence, token) per position, as Fast-dLLM's samplers compute them."""
    if arch == "llada":  # Gumbel-max in float64, confidence = softmax prob of the choice
        if temperature > 0:
            lg = logits.to(torch.float64)
            noise = torch.rand_like(lg)
            x0 = torch.argmax(lg.exp() / (-torch.log(noise)) ** temperature, dim=-1)
        else:
            x0 = torch.argmax(logits, dim=-1)
        p = F.softmax(logits.to(torch.float64), dim=-1)
        return torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1), x0
    lg = logits.float()  # Dream: temperature / top-p sampling, confidence = prob
    if temperature > 0:
        lg = lg / temperature
    if top_p is not None and top_p < 1:
        lg = _top_p_logits(lg, top_p)
    probs = torch.softmax(lg, dim=-1)
    if temperature > 0:
        x0 = torch.distributions.Categorical(probs=probs).sample()
        conf = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        conf, x0 = probs.max(dim=-1)
    return conf, x0


def select(conf, mask, quota=None, threshold=None):
    """Positions to commit: top-`quota` (fixed) or all >= threshold plus the best one."""
    neg = torch.finfo(conf.dtype).min
    conf = torch.where(mask, conf, torch.full_like(conf, neg))
    n_masked = int(mask.sum())
    if n_masked == 0:
        return torch.zeros_like(mask)
    if threshold is not None:
        pick = mask & (conf >= threshold)
        pick[0, int(torch.argmax(conf[0]))] = True
        return pick & mask
    k = max(1, min(int(quota), n_masked))
    pick = torch.zeros_like(mask)
    pick[0, torch.topk(conf[0], k).indices] = True
    return pick & mask


def tokens_per_step(n_masked, steps):
    """Uniform schedule: how many tokens each of `steps` steps commits."""
    base, rem = divmod(int(n_masked), int(steps))
    return [base + (1 if i < rem else 0) for i in range(steps)]


# --------------------------------------------------------------------------- generation
@torch.no_grad()
def kv_generate(model, arch, tokenizer, ids, gen_length, block_length, seed, *,
                cache="dual", decode="threshold", threshold=0.9, steps_per_block=None,
                temperature=0.0, top_p=None, link=None, link_cfg=None, device=None):
    """One generation with Fast-dLLM's block decoding.

    cache   "dual" (Fast-dLLM DualCache) or "none" (full forward every step)
    decode  "threshold" (parallel decoding, gamma = threshold) or "fixed"
            (steps_per_block steps per block; default block_length = 1 token/step)
    Dream's logit shift: as in Fast-dLLM, the first token of a block is committed
    from the block-start full forward; inside the block the logits are shifted
    within the chunk, so position s+j is read from row s+j-1.
    Returns generated ids, number of steps (= model forwards = round trips),
    commits per step, wall time and the link's per-step statistics.
    """
    if cache not in {"dual", "none"} or decode not in {"threshold", "fixed"}:
        raise ValueError(f"cache={cache!r}, decode={decode!r}")
    if gen_length % block_length:
        raise ValueError("gen_length must be a multiple of block_length")
    steps_per_block = int(steps_per_block or block_length)
    device = torch.device(device) if device is not None else ids.device
    edge = link.edge if link is not None else device
    mask_id = mask_token_id(arch, tokenizer)
    lp = int(ids.shape[1])
    n = lp + int(gen_length)
    x = torch.full((1, n), mask_id, dtype=torch.long, device=device)
    x[:, :lp] = ids.to(device)
    commits = []

    def forward(lo, hi, kind, block, pkv=None, rp=None, use_cache=False):
        if link is not None:
            link.set_chunk(lo, kind, block)
        inp = x[:, lo:hi]
        if arch == "dream":
            if pkv is None:
                out = model(inp, "full", None, use_cache=use_cache)
            else:
                out = model(inp, "full", None, past_key_values=pkv, use_cache=True,
                            dual_cache=True, replace_position=rp)
        else:
            if pkv is None:
                out = model(inp, use_cache=use_cache)
            else:
                out = model(inp, past_key_values=pkv, use_cache=True, replace_position=rp)
        logits = out.logits
        if arch == "dream":  # Dream predicts position i from row i-1 (within this chunk)
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return logits, (out.past_key_values if use_cache else None)

    def apply(s, e, x0, pick):
        x[:, s:e] = torch.where(pick, x0.to(x.dtype), x[:, s:e])
        n_new = int(pick.sum())
        commits.append(n_new)
        if link is not None:
            link.end_step(n_new)

    set_seed(int(seed))
    if link is not None:
        link.begin(n_total=n, **(link_cfg or {}))
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    try:
        for b in range(int(gen_length) // int(block_length)):
            s, e = lp + b * block_length, lp + (b + 1) * block_length
            quota = tokens_per_step(int((x[:, s:e] == mask_id).sum()), steps_per_block)
            i = 0
            if cache == "dual":
                logits, pkv = forward(0, n, "full", b, use_cache=True)
                conf, x0 = propose(arch, logits[:, s:e], temperature, top_p)
                mask_b = x[:, s:e] == mask_id
                if arch == "dream":  # first block token from the correctly shifted full logits
                    pick = torch.zeros_like(mask_b)
                    pick[0, 0] = bool(mask_b[0, 0])
                    if not pick.any():
                        pick = select(conf, mask_b, quota=1)
                else:
                    pick = select(conf, mask_b, quota=quota[0] if decode == "fixed" else None,
                                  threshold=threshold if decode == "threshold" else None)
                apply(s, e, x0, pick)
                i = 1
                rp = torch.zeros((1, n), dtype=torch.bool, device=edge)
                rp[:, s:e] = True
                while bool((x[:, s:e] == mask_id).any()):
                    logits, _ = forward(s, e, "block", b, pkv=pkv, rp=rp)
                    conf, x0 = propose(arch, logits, temperature, top_p)
                    mask_b = x[:, s:e] == mask_id
                    q = quota[i] if i < len(quota) else 1
                    pick = select(conf, mask_b, quota=q if decode == "fixed" else None,
                                  threshold=threshold if decode == "threshold" else None)
                    apply(s, e, x0, pick)
                    i += 1
            else:
                while bool((x[:, s:e] == mask_id).any()):
                    logits, _ = forward(0, n, "full", b)
                    conf, x0 = propose(arch, logits[:, s:e], temperature, top_p)
                    mask_b = x[:, s:e] == mask_id
                    q = quota[i] if i < len(quota) else 1
                    pick = select(conf, mask_b, quota=q if decode == "fixed" else None,
                                  threshold=threshold if decode == "threshold" else None)
                    apply(s, e, x0, pick)
                    i += 1
    except BaseException:
        if link is not None:
            link.active = False
        raise
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    split_steps = link.end() if link is not None else None
    return {"ms": elapsed_ms, "generated_ids": x[0, lp:].detach().cpu().long().clone(),
            "n_steps": len(commits), "commits": commits, "split_steps": split_steps}
