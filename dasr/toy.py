"""Tiny bidirectional transformer; semantic debugging only, no benchmark claims."""
import numpy as np


class ToyModel:
    family = "toy"
    logical_dtype = "float32"

    def __init__(self, seed=7, layers=4, width=24, vocab=64):
        self.layers, self.width, self.vocab, self.mask_id = layers, width, vocab, 0
        rng = np.random.RandomState(seed)
        self.emb = rng.normal(0, .2, (vocab, width)).astype(np.float32)
        self.weights = [{name: rng.normal(0, .1, (width, width)).astype(np.float32)
                         for name in ("q", "k", "v", "o", "f1", "f2")} for _ in range(layers)]
        self.head = rng.normal(0, .1, (width, vocab)).astype(np.float32)
        self.reset()

    def reset(self):
        self.ffn_cache = {}
        self.selected_rows = 0
        self.total_rows = 0
        self.audit_refreshes = 0
        self.selected_by_layer = {i: 0 for i in range(self.layers)}

    def blocks(self, h, start, stop, active=None, anchor=True, audit_threshold=.1):
        for i in range(start, stop):
            w = self.weights[i]
            q, k, v = h @ w["q"], h @ w["k"], h @ w["v"]
            scores = (q @ k.T) / np.sqrt(self.width)
            scores -= scores.max(axis=1, keepdims=True)
            attn = np.exp(scores)
            attn /= attn.sum(axis=1, keepdims=True)
            h = h + (attn @ v) @ w["o"]
            indices = np.arange(len(h)) if anchor or active is None or i not in self.ffn_cache else np.asarray(active)
            out = np.zeros_like(h) if i not in self.ffn_cache else self.ffn_cache[i].copy()
            out[indices] = np.tanh(h[indices] @ w["f1"]) @ w["f2"]
            self.selected_rows += len(indices)
            self.selected_by_layer[i] += len(indices)
            self.total_rows += len(h)
            self.ffn_cache[i] = out
            h = h + out
        return h

    def prefix(self, ids, split, active=None, anchor=True):
        return self.blocks(self.emb[np.asarray(ids)], 0, split, active, anchor)

    def suffix(self, hidden, split, active=None, anchor=True):
        return self.blocks(hidden, split, self.layers, active, anchor) @ self.head

    def forward(self, ids):
        return self.suffix(self.prefix(ids, 0), 0)

    def proposals(self, logits):
        # Suppress the toy MASK token to guarantee a complete debug trajectory.
        logits = logits.copy()
        logits[:, self.mask_id] = -1e9
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        ids = logits.argmax(axis=1)
        return ids.astype(np.int64), probs[np.arange(len(ids)), ids]

    def to_wire(self, hidden):
        return np.ascontiguousarray(hidden), self.logical_dtype

    def from_wire(self, hidden, logical_dtype):
        return hidden

    def sync(self, side=None):
        pass

    def memory(self):
        return dict(edge_peak_allocated_bytes=None, cloud_peak_allocated_bytes=None)
