"""Extensions around the OFFICIAL dLLM-Cache Dream hook (Exp10B/Exp10C).

Nothing in the official checkout is edited. Everything is patched at runtime and
restored on exit, because ``dLLMCache`` is a process-wide singleton: an attribute
left on the instance would leak into the next run (incl. "baseline" runs that
reuse the same Python process).

1. Prompt-tail fix (``prompt_tail_rows``)
   Dream shifts logits by one position:
       logits = cat([logits[:, :1], logits[:, :-1]], dim=1)
   so response token 0 is predicted from the hidden state of the LAST PROMPT
   ROW. The official hook splits at ``prompt_length``, which puts that row in
   the prompt cache (refreshed only every Kp steps). With Kp=100 the first answer
   token is decided from step-1 features (response all [MASK]).
   ``prompt_tail_rows=B`` calls ``reset_cache(prompt_len - B)`` so the last B prompt
   rows belong to the response cache. With ``force_prompt_tail=True`` those rows are
   also always included in the partial-update index (transfer_ratio) on
   non-refresh steps.

2. Refresh decisions come from ``RefreshController`` (fixed or rule-based
   adaptive), installed as instance-level overrides of ``refresh_gen`` /
   ``refresh_prompt``. Decisions are logged for both kinds of policy.

3. ``StepObserver`` is passed to Dream's ``generation_tokens_hook_func``. It sees
   the tokens committed at every step (+ their probability / top1-top2 margin)
   and feeds the controller.
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict

import torch

from src.refresh_rules import RefreshController, count_frontier_eos, make_policy
from src.utils import set_seed

HOOK_MODULE = "dllm_cache.hooks.cache_hook_Dream"
_PATCHED_ATTRS = ("refresh_gen", "refresh_prompt")


def special_ids(tokenizer, model):
    """mask id + ids that count as 'no content' (EOS / pad / <|im_end|>)."""
    mask_id = getattr(model.generation_config, "mask_token_id", None)
    if mask_id is None:
        mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError("Cannot determine Dream mask_token_id")
    non_content = set()
    for value in (tokenizer.eos_token_id, tokenizer.pad_token_id,
                  getattr(model.generation_config, "eos_token_id", None),
                  getattr(model.generation_config, "pad_token_id", None)):
        if isinstance(value, int):
            non_content.add(value)
        elif isinstance(value, (list, tuple)):
            non_content.update(int(v) for v in value)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tokenizer.unk_token_id:
        non_content.add(im_end)
    return int(mask_id), non_content


class StepObserver:
    """Dream ``generation_tokens_hook_func(step, x, logits)``.

    ``step`` is the 0-based loop index i; it corresponds to history[i] and to
    dLLMCache step i + 1. Commit steps are reported 1-based (i + 1) to match
    ``compare_history(...)["first_divergent_step"]``.
    """

    def __init__(self, prompt_len, mask_id, non_content_ids, controller=None,
                 log_confidence=True):
        self.plen = int(prompt_len)
        self.mask_id = int(mask_id)
        self.non_content = torch.tensor(sorted(non_content_ids) or [-1])
        self.controller = controller
        self.log_confidence = bool(log_confidence)
        self.prev_masked = None
        self.max_content_pos = -1  # rightmost committed content position
        window = getattr(getattr(controller, "policy", None), "eos_frontier_window", 8)
        self.frontier_window = int(window)
        self.commits = []  # dicts: step, pos, token, prob, margin, frontier_eos

    def __call__(self, step, x, logits):
        if step is None:
            return x
        gen = x[0, self.plen:]
        now_masked = gen == self.mask_id
        if self.prev_masked is None:
            self.prev_masked = torch.ones_like(now_masked)
        newly = self.prev_masked & ~now_masked
        self.prev_masked = now_masked
        pos = newly.nonzero(as_tuple=True)[0]
        n_new = int(pos.numel())
        n_content, n_eos, n_frontier, min_prob = 0, 0, 0, None
        frontier_flags = []
        if n_new:
            toks = gen[pos]
            if self.non_content.device != toks.device:
                self.non_content = self.non_content.to(toks.device)
            is_nc = torch.isin(toks, self.non_content)
            n_eos = int(is_nc.sum())
            n_content = n_new - n_eos
            pos_list = pos.tolist()
            content_list = (~is_nc).tolist()
            n_frontier, self.max_content_pos = count_frontier_eos(
                self.max_content_pos, pos_list, content_list, self.frontier_window)
            frontier_flags = [self.frontier_window > 0 and (not c)
                              and p <= self.max_content_pos + self.frontier_window
                              for p, c in zip(pos_list, content_list)]
            want_conf = self.log_confidence or (
                self.controller is not None
                and getattr(self.controller.policy, "low_conf_threshold", 0.0) > 0)
            if (want_conf and logits is not None and logits.dim() == 3
                    and logits.shape[1] == x.shape[1]):
                probs = logits[0, self.plen + pos].float().softmax(-1)
                top2 = probs.topk(2, dim=-1).values
                p_tok = probs.gather(-1, toks.view(-1, 1)).squeeze(-1)
                min_prob = float(p_tok.min())
                if self.log_confidence:
                    for p_i, t_i, pr, m, fr in zip(pos_list, toks.tolist(), p_tok.tolist(),
                                                   (top2[:, 0] - top2[:, 1]).tolist(),
                                                   frontier_flags):
                        self.commits.append({"step": int(step) + 1, "pos": p_i, "token": t_i,
                                             "prob": pr, "margin": m, "frontier_eos": fr})
            elif self.log_confidence:
                for p_i, t_i, fr in zip(pos_list, toks.tolist(), frontier_flags):
                    self.commits.append({"step": int(step) + 1, "pos": p_i, "token": t_i,
                                         "prob": None, "margin": None, "frontier_eos": fr})
        if self.controller is not None:
            self.controller.observe(int(step) + 1, n_content, n_eos, min_prob,
                                    n_frontier_eos=n_frontier)
        return x


@contextmanager
def feature_cache_session(model, api, prompt_len, policy, prompt_tail_rows=0,
                          force_prompt_tail=True):
    """Register the official Dream hook with a controlled refresh plan."""
    dLLMCache, dLLMCacheConfig, register, logout = api
    policy = make_policy(policy)
    prompt_tail_rows = int(prompt_tail_rows)
    split = int(prompt_len) - prompt_tail_rows
    if prompt_tail_rows < 0 or split < 1:
        raise ValueError(f"Invalid prompt_tail_rows={prompt_tail_rows} for prompt_len={prompt_len}")
    controller = RefreshController(policy)

    kp = policy.prompt_interval if policy.kind == "fixed" else 1
    kr = policy.gen_interval if policy.kind == "fixed" else 1
    inst = dLLMCache.new_instance(**asdict(dLLMCacheConfig(
        prompt_interval_steps=int(kp), gen_interval_steps=int(kr),
        transfer_ratio=float(policy.transfer_ratio),
    )))
    for attr in _PATCHED_ATTRS:  # defensive: remove leftovers of a crashed run
        inst.__dict__.pop(attr, None)

    def refresh_gen(layer_id=0):
        return controller.decide(inst.current_step)[0]

    def refresh_prompt(layer_id=0):
        return controller.decide(inst.current_step)[1]

    inst.refresh_gen = refresh_gen
    inst.refresh_prompt = refresh_prompt
    controller.step_fn = lambda: inst.current_step  # used by the split runtime

    hook_mod = sys.modules[HOOK_MODULE]
    original_refresh_index = hook_mod.refresh_index
    if prompt_tail_rows > 0 and force_prompt_tail:
        def refresh_index(new_features, cached_features=None, transfer_ratio=0.5, layer_id=0):
            _, gen_len, _ = new_features.shape
            k = int(gen_len * transfer_ratio)
            cos = torch.nn.functional.cosine_similarity(new_features, cached_features, dim=-1)
            cos[:, :prompt_tail_rows] = -2.0  # cosine >= -1, so prompt-tail rows always selected
            return torch.topk(cos, largest=False, k=k).indices
        hook_mod.refresh_index = refresh_index

    registered = False
    try:
        register(model, "model.layers")
        registered = True
        inst.reset_cache(split)
        yield controller
    finally:
        if registered:
            logout(model, "model.layers")
        hook_mod.refresh_index = original_refresh_index
        for attr in _PATCHED_ATTRS:
            inst.__dict__.pop(attr, None)


def generate(model, tokenizer, ids, mask, new_tokens, steps, seed, *, alg,
             temperature, top_p=None, top_k=None, alg_temp=0.0,
             cache_api=None, policy=None, prompt_tail_rows=0, force_prompt_tail=True,
             capture_trajectory=True, log_confidence=True, observe=True,
             link=None, link_cfg=None):
    """One Dream generation (baseline when cache_api is None).

    observe=False skips the token hook (pure timing runs). It is forced on for
    adaptive policies, because the hook is part of the policy itself.
    link: optional src.split_runtime.SplitLink (two-GPU split); link_cfg holds its
    codec settings (rows / encoding / anchor_interval). Per-step split statistics
    are returned under "split_steps".
    """
    plen = int(ids.shape[1])
    mask_id, non_content = special_ids(tokenizer, model)
    device = ids.device
    if cache_api is not None and make_policy(policy).kind == "adaptive":
        observe = True
    set_seed(int(seed))
    session = (feature_cache_session(model, cache_api, plen, policy, prompt_tail_rows,
                                     force_prompt_tail)
               if cache_api is not None else nullcontext(None))
    with session as controller:
        observer = StepObserver(plen, mask_id, non_content, controller,
                                log_confidence and observe)
        kwargs = dict(max_new_tokens=int(new_tokens), steps=int(steps),
                      temperature=temperature, top_p=top_p, top_k=top_k, alg=alg,
                      output_history=bool(capture_trajectory),
                      return_dict_in_generate=True)
        if observe:
            kwargs["generation_tokens_hook_func"] = observer
        if alg != "origin":
            kwargs["alg_temp"] = alg_temp
        if link is not None:
            link.begin(prompt_len=plen, controller=controller,
                       prompt_tail_rows=prompt_tail_rows if controller is not None else 0,
                       mask_id=mask_id, **(link_cfg or {}))
        set_seed(int(seed))  # after cache setup, exactly like Exp9B/10A
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                out = model.diffusion_generate(ids, attention_mask=mask, **kwargs)
        except BaseException:
            if link is not None:
                link.active = False
            raise
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        split_steps = link.end() if link is not None else None
        refresh = controller.summary() if controller is not None else None

    generated = out.sequences[0, plen:].detach().cpu().long().clone()
    history = None
    if capture_trajectory:
        raw = getattr(out, "history", None)
        if raw is None or len(raw) != int(steps):
            raise RuntimeError(f"Expected {steps} history states, got "
                               f"{None if raw is None else len(raw)}")
        history = [h[0].detach().cpu().long().clone() for h in raw]
        if not torch.equal(history[-1][plen:], generated):
            raise RuntimeError("Last history state does not match final output")
    return {"ms": elapsed_ms, "generated_ids": generated, "history": history,
            "commits": observer.commits, "refresh": refresh, "split_steps": split_steps}


def first_commit_step(commits, pos):
    for c in commits:
        if c["pos"] == pos:
            return c
    return None


def divergence_detail(base, cached, prompt_len):
    """Where/why the cached trajectory first left the baseline trajectory."""
    hb, hc = base["history"], cached["history"]
    result = {"div_positions": None, "div_includes_pos0": None,
              "div_first_pos": None, "div_base_token": None, "div_cache_token": None,
              "div_base_commit_prob": None, "div_base_commit_margin": None,
              "div_cache_commit_prob": None}
    for i, (a, b) in enumerate(zip(hb, hc)):
        if torch.equal(a, b):
            continue
        diff = (a[prompt_len:] != b[prompt_len:]).nonzero(as_tuple=True)[0].tolist()
        result["div_positions"] = diff[:16]
        result["div_includes_pos0"] = 0 in diff
        if diff:
            p = diff[0]
            result["div_first_pos"] = p
            result["div_base_token"] = int(a[prompt_len + p])
            result["div_cache_token"] = int(b[prompt_len + p])
            cb = first_commit_step(base["commits"], p)
            cc = first_commit_step(cached["commits"], p)
            if cb:
                result["div_base_commit_prob"] = cb["prob"]
                result["div_base_commit_margin"] = cb["margin"]
            if cc:
                result["div_cache_commit_prob"] = cc["prob"]
        break
    pos0_b = first_commit_step(base["commits"], 0)
    pos0_c = first_commit_step(cached["commits"], 0)
    result["pos0_commit_step_base"] = pos0_b["step"] if pos0_b else None
    result["pos0_commit_step_cache"] = pos0_c["step"] if pos0_c else None
    result["first_token_equal"] = bool(base["generated_ids"][0] == cached["generated_ids"][0])
    return result
