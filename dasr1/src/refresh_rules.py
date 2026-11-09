"""Rule-based refresh decisions for dLLM-Cache feature caching (pure Python).

Used by Exp10B/Exp10C. No torch import, so the rules can be unit-tested anywhere.

Step numbering follows ``dLLMCache.current_step`` (1-based: the first forward
pass of diffusion_generate is step 1). The official Dream hook asks
``refresh_gen()`` / ``refresh_prompt()`` once per layer per step; the controller
answers once per step and memoizes, so every layer of a step sees the same plan.

After step ``s`` has committed its tokens, ``observe(s, ...)`` is called by the
token hook. The decision for step ``s + 1`` therefore uses every token committed
up to and including step ``s`` (exactly what a refresh at ``s + 1`` would see).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class FixedPolicy:
    """Official dLLM-Cache schedule: refresh every Kp / Kr steps."""
    prompt_interval: int
    gen_interval: int
    transfer_ratio: float
    kind: str = "fixed"

    def label(self) -> str:
        return f"fixed(Kp={self.prompt_interval},Kr={self.gen_interval},rho={self.transfer_ratio})"


@dataclass(frozen=True)
class AdaptivePolicy:
    """Content-commit-triggered refresh (rule-based, no extra forward pass).

    Rationale (RQ1): a single committed token changes hidden states densely, so
    staleness grows with the number of *meaningful* commits since the last
    refresh, not with wall-clock steps. Meaningful = content tokens, plus EOS
    tokens committed near the end of the answer ("frontier EOS"), because those
    decide the answer length. Exp10B showed that stale features cause exactly
    such early EOS commits (answers cut to "1." or emptied). EOS commits far out
    in the padding tail are not counted.

    A commit counts as frontier EOS when it is an EOS/pad token at a position
    <= (rightmost committed content position) + eos_frontier_window. Set the
    window to 0 to count content tokens only.

    Rules, evaluated once per step s > 1 ("trigger" = content + frontier EOS):
      prompt refresh  if triggers since last prompt refresh >= prompt_content_commits
                      or steps since last prompt refresh  >= prompt_max_age
      gen refresh     if a prompt refresh happens (full refresh)
                      or triggers since last gen refresh >= gen_content_commits
                      or steps since last gen refresh >= gen_max_age
                      or (low_conf_threshold > 0 and the previous step committed a
                          token whose probability was below the threshold)

    The max ages are the preventive part (Exp10B: errors pile up at the oldest
    step of the cache window); the commit triggers are reactive (they stop an
    error from cascading into the next commits).

    With gen_content_commits = prompt_content_commits = a huge number and
    max ages Kr / Kp (Kp a multiple of Kr), this reduces exactly to
    FixedPolicy(Kp, Kr) -- a useful sanity check / ablation.
    """
    gen_content_commits: int = 2
    gen_max_age: int = 4
    prompt_content_commits: int = 16
    prompt_max_age: int = 32
    eos_frontier_window: int = 8
    low_conf_threshold: float = 0.0
    transfer_ratio: float = 0.25
    kind: str = "adaptive"

    def label(self) -> str:
        return (f"adaptive(G={self.gen_content_commits},Ag={self.gen_max_age},"
                f"P={self.prompt_content_commits},Ap={self.prompt_max_age},"
                f"W={self.eos_frontier_window},conf={self.low_conf_threshold},"
                f"rho={self.transfer_ratio})")


def count_frontier_eos(max_content_pos, positions, is_content, window):
    """Count EOS commits near the answer end for one step.

    max_content_pos: rightmost committed content position before this step (-1 if none).
    positions / is_content: this step's newly committed response positions.
    Returns (n_frontier_eos, updated max_content_pos). Content committed in the
    same step is taken into account first.
    """
    new_max = max([max_content_pos] + [p for p, c in zip(positions, is_content) if c])
    if window <= 0:
        return 0, new_max
    n = sum(1 for p, c in zip(positions, is_content) if not c and p <= new_max + window)
    return n, new_max


def make_policy(spec):
    """Build a policy from config: ("fixed", Kp, Kr, rho) or ("adaptive", {kwargs})."""
    if isinstance(spec, (FixedPolicy, AdaptivePolicy)):
        return spec
    kind = spec[0]
    if kind == "fixed":
        _, kp, kr, rho = spec
        if int(kp) < 1 or int(kr) < 1 or not 0 <= float(rho) <= 1:
            raise ValueError(f"Invalid fixed policy {spec}")
        return FixedPolicy(int(kp), int(kr), float(rho))
    if kind == "adaptive":
        params = dict(spec[1]) if len(spec) > 1 else {}
        policy = AdaptivePolicy(**params)
        if min(policy.gen_content_commits, policy.gen_max_age,
               policy.prompt_content_commits, policy.prompt_max_age) < 1:
            raise ValueError(f"Adaptive thresholds must be >= 1: {spec}")
        if policy.eos_frontier_window < 0:
            raise ValueError(f"eos_frontier_window must be >= 0: {spec}")
        return policy
    raise ValueError(f"Unknown policy kind {kind!r}")


class RefreshController:
    def __init__(self, policy):
        self.policy = make_policy(policy)
        self.decisions: dict[int, tuple[bool, bool]] = {}
        self.reasons: Counter = Counter()
        self._last_gen = 0
        self._last_prompt = 0
        self._content_since_gen = 0
        self._content_since_prompt = 0
        self._low_conf_pending = False
        self.observed_steps = 0
        self.total_content_commits = 0
        self.total_eos_commits = 0
        self.total_frontier_eos = 0

    # -- decisions ---------------------------------------------------------
    def decide(self, step: int) -> tuple[bool, bool]:
        """Return (refresh_gen, refresh_prompt) for this step; memoized."""
        step = int(step)
        if step in self.decisions:
            return self.decisions[step]
        p = self.policy
        if step <= 1:
            gen = prompt = True
            self.reasons["init"] += 1
        elif p.kind == "fixed":
            prompt = (step - 1) % p.prompt_interval == 0
            gen = (step - 1) % p.gen_interval == 0
            if prompt:
                self.reasons["prompt_interval"] += 1
            if gen:
                self.reasons["gen_interval"] += 1
        else:
            prompt = False
            if self._content_since_prompt >= p.prompt_content_commits:
                prompt = True
                self.reasons["prompt_content"] += 1
            elif step - self._last_prompt >= p.prompt_max_age:
                prompt = True
                self.reasons["prompt_age"] += 1
            gen = prompt
            if not gen:
                if self._content_since_gen >= p.gen_content_commits:
                    gen = True
                    self.reasons["gen_content"] += 1
                elif self._low_conf_pending:
                    gen = True
                    self.reasons["gen_low_conf"] += 1
                elif step - self._last_gen >= p.gen_max_age:
                    gen = True
                    self.reasons["gen_age"] += 1
        if gen:
            self._last_gen = step
            self._content_since_gen = 0
            self._low_conf_pending = False
        if prompt:
            self._last_prompt = step
            self._content_since_prompt = 0
        self.decisions[step] = (bool(gen), bool(prompt))
        return self.decisions[step]

    # -- observations from the token hook -----------------------------------
    def observe(self, step: int, n_content: int, n_eos: int, min_prob: float | None = None,
                n_frontier_eos: int = 0) -> None:
        """n_eos counts all EOS/pad commits; n_frontier_eos is the subset near
        the answer end (see AdaptivePolicy). Only content + frontier EOS trigger."""
        self.observed_steps = max(self.observed_steps, int(step))
        window = getattr(self.policy, "eos_frontier_window", 0)
        triggers = int(n_content) + (int(n_frontier_eos) if window > 0 else 0)
        self._content_since_gen += triggers
        self._content_since_prompt += triggers
        self.total_content_commits += int(n_content)
        self.total_eos_commits += int(n_eos)
        self.total_frontier_eos += int(n_frontier_eos)
        thr = getattr(self.policy, "low_conf_threshold", 0.0)
        if thr > 0 and min_prob is not None and min_prob < thr:
            self._low_conf_pending = True

    # -- reporting ----------------------------------------------------------
    def summary(self) -> dict:
        steps = len(self.decisions)
        n_gen = sum(g for g, _ in self.decisions.values())
        n_prompt = sum(p for _, p in self.decisions.values())
        return {
            "policy": self.policy.label(),
            "decided_steps": steps,
            "gen_refreshes": n_gen,
            "prompt_refreshes": n_prompt,
            "gen_refresh_frac": n_gen / steps if steps else None,
            "prompt_refresh_frac": n_prompt / steps if steps else None,
            "content_commits": self.total_content_commits,
            "eos_commits": self.total_eos_commits,
            "frontier_eos_commits": self.total_frontier_eos,
            "refresh_reasons": dict(self.reasons),
            "gen_refresh_steps": [s for s, (g, _) in sorted(self.decisions.items()) if g],
            "prompt_refresh_steps": [s for s, (_, p) in sorted(self.decisions.items()) if p],
        }
