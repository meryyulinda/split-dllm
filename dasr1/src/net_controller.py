"""Exp18: a controller that picks the boundary codec (and Fast-dLLM's gamma) from the link.

Attach as `link.codec_ctl = CodecController(...)` and run the link with
encoding="adaptive"; for Fast-dLLM pass `threshold=ctl.gamma` to kv_generate.

Measurements (wall clock, from the real link of src/net_wire.py):
  throughput  B   = uplink bytes / (uplink time - RTT/2 - host copy time), smoothed
                    (EWMA, alpha); the prototype's edge and cloud share a clock, a real
                    deployment would have the cloud report its receive time
  compute     c   = time per step that is not uplink or downlink (edge, codec,
                    cloud, sampler, emulated edge slowdown), smoothed (EWMA)
  per step:   t(codec) = c + RTT + V(codec) / B,   V = rows x 2d (bf16) or rows x (d + 2)

Codec rule (lam, hysteresis h):
  switch bf16 -> INT8  when INT8 would shorten the step by more than lam:
                       (V_bf16 - V_int8) / B  >  lam * t(bf16)
  switch INT8 -> bf16  when the saving falls below lam / h (default h = 1: no
                       hysteresis; switching costs nothing, both sides always hold
                       the reference, so the codec can follow every step).
  The saving depends on the step's rows, so a large message (a block-start or
  prompt-refresh step) can go as INT8 while the small ones of the same link stay
  bf16. bf16 rows are exact; every INT8 step makes the answer approximate, so the
  rule keeps bf16 whenever INT8 would not save a noticeable share of the time.
Gamma rule (adapt_gamma, Fast-dLLM): gamma_low while the link takes more than
  `net_share` of the step time, (RTT + V/B) / t > net_share, back to gamma_high
  below net_share / h. Fewer steps then save round trips and bytes.
Modes:
  "per_step"     decide at every step
  "per_request"  decide once, from the first `warmup` steps, then keep the choice
                 (what the backend selector does today: one codec per request)
Before the first measurements the controller sends bf16 and uses gamma_high.
"""
from __future__ import annotations

import time


class CodecController:
    def __init__(self, mode="per_step", rtt_ms=20.0, lam=0.10, hysteresis=1.0, alpha=0.3,
                 adapt_gamma=False, gamma_high=0.9, gamma_low=0.7, net_share=0.5, warmup=2,
                 host_ms=1.0):
        if mode not in {"per_step", "per_request"}:
            raise ValueError(f"mode={mode!r}")
        self.mode, self.rtt_ms = mode, float(rtt_ms)
        self.lam, self.hyst, self.alpha = float(lam), float(hysteresis), float(alpha)
        self.adapt_gamma = bool(adapt_gamma)
        self.g_hi, self.g_lo, self.net_share = float(gamma_high), float(gamma_low), float(net_share)
        self.warmup = int(warmup)
        self.host_ms = float(host_ms)   # GPU<->host copies and sockets per uplink (Exp16: ~1-1.5 ms)
        self.reset()

    # ----------------------------------------------------------------- state
    def reset(self):
        self.bw = None            # bytes per ms
        self.comp = None          # ms per step outside the link
        self.vbytes = None        # bytes per uplink (EWMA), for the gamma rule
        self.codec = "bf16"
        self.g = self.g_hi
        self.frozen = False
        self.n_up = 0
        self.t_last = None        # wall clock at the end of the previous uplink
        self.down_ms = 0.0        # downlink time since the previous uplink
        self.switches = 0
        self.log = []             # (step, codec, gamma, bw Mbit/s, comp ms)

    def _ewma(self, old, new):
        return new if old is None else (1 - self.alpha) * old + self.alpha * new

    def observe_down(self, info):
        self.down_ms += float(info.get("down_wire_ms", 0.0))

    def observe_up(self, info, d=None):
        now = time.perf_counter()
        up_ms = float(info["up_wire_ms"])
        nbytes = float(info["wire_up_bytes"])
        transfer = max(up_ms - self.rtt_ms / 2 - self.host_ms, 0.05)
        self.bw = self._ewma(self.bw, nbytes / transfer)
        self.vbytes = self._ewma(self.vbytes, nbytes)
        if self.t_last is not None:   # one full step: previous uplink end -> this uplink end
            comp = (now - self.t_last) * 1e3 - up_ms - self.down_ms
            self.comp = self._ewma(self.comp, max(comp, 0.0))
        self.t_last, self.down_ms = now, 0.0
        self.n_up += 1

    # ----------------------------------------------------------------- decisions
    def _ready(self):
        return self.bw is not None and self.comp is not None

    def choose(self, step, n_rows, d):
        if self.frozen or not self._ready():
            return self.codec
        v16, v8 = n_rows * 2 * d, n_rows * (d + 2)
        t16 = self.comp + self.rtt_ms + v16 / self.bw
        saving = (v16 - v8) / self.bw / t16
        new = self.codec
        if self.codec == "bf16" and saving > self.lam:
            new = "int8_delta"
        elif self.codec == "int8_delta" and saving < self.lam / self.hyst:
            new = "bf16"
        if new != self.codec:
            self.switches += 1
            self.codec = new
        if self.adapt_gamma:
            share = (self.rtt_ms + self.vbytes / self.bw) / (
                self.comp + self.rtt_ms + self.vbytes / self.bw)
            if self.g == self.g_hi and share > self.net_share:
                self.g = self.g_lo
            elif self.g == self.g_lo and share < self.net_share / self.hyst:
                self.g = self.g_hi
        self.log.append((int(step), self.codec, self.g, self.bw * 8 / 1e3, self.comp))
        if self.mode == "per_request" and self.n_up >= self.warmup:
            self.frozen = True
        return self.codec

    def gamma(self):
        return self.g

    def summary(self):
        return {"ctl_switches": self.switches,
                "ctl_final_codec": self.codec,
                "ctl_bw_mbps_last": None if self.bw is None else self.bw * 8 / 1e3,
                "ctl_comp_ms_last": self.comp}
