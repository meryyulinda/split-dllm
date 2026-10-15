"""Measured table scheduler. Calibration risk is empirical, not a formal bound."""
from dataclasses import dataclass


@dataclass
class Action:
    name: str
    split: int
    codec: str
    threshold: float
    refresh: int
    edge_ms: float
    cloud_ms: float
    encode_ms: float
    decode_ms: float
    uplink_bytes: float
    downlink_bytes: float
    memory_bytes: float
    risk: float
    switch_ms: float = 0.
    selective: bool = False
    active_fraction: float = .25

    def cost(self, bandwidth_mbps, rtt_ms):
        if bandwidth_mbps <= 0 or rtt_ms < 0:
            raise ValueError("Invalid network state")
        return (self.edge_ms + self.cloud_ms + self.encode_ms + self.decode_ms + rtt_ms
                + 8 * (self.uplink_bytes + self.downlink_bytes) / (bandwidth_mbps * 1000))


class Scheduler:
    def __init__(self, actions, memory_budget, risk_budget, hysteresis=.05, horizon=4):
        if horizon < 1 or not 0 <= hysteresis < 1:
            raise ValueError("Invalid scheduler settings")
        self.actions = [a for a in actions if a.memory_bytes <= memory_budget and a.risk <= risk_budget]
        if not self.actions:
            raise ValueError("No configuration passes measured memory and quality constraints")
        self.hysteresis, self.horizon = hysteresis, horizon
        self.current = None

    def choose(self, bandwidth_mbps, rtt_ms, anchor=False):
        if self.current is not None and not anchor:
            return self.current
        costs = {a.name: a.cost(bandwidth_mbps, rtt_ms) +
                 (a.switch_ms / self.horizon if self.current is not None and a != self.current else 0)
                 for a in self.actions}
        best = min(self.actions, key=lambda a: costs[a.name])
        if self.current is not None and costs[best.name] >= self.current.cost(bandwidth_mbps, rtt_ms) * (1 - self.hysteresis):
            best = self.current
        self.current = best
        return best
