"""LLaDA-8B-Instruct with the OFFICIAL dLLM-Cache LLaDA hook (RQ5, Exp12).

Mirrors ``src/feature_cache_ext.generate`` (Dream) so the same experiment code
(Exp10-style grid, Exp11 split prototype) can run on LLaDA:

  * Sampler = official LLaDA ``generate.py``: semi-autoregressive blocks of
    ``block_length`` tokens, ``low_confidence`` remasking, Gumbel-max sampling in
    float64 when temperature > 0, no classifier-free guidance.
  * Feature cache = official ``register_cache_LLaDA`` on
    ``model.transformer.blocks``, refresh decisions from ``RefreshController``
    (fixed Kp/Kr or the rule-based adaptive policy), exactly like Dream.
  * NO prompt-tail correction: LLaDA predicts token i from hidden row i (no logit
    shift), so the official split at ``prompt_len`` is already right
    (prompt_tail_rows = 0).
  * ``link`` = optional ``src.split_runtime.SplitLink`` (two-GPU split).

Returned dict = same keys as the Dream version: ms, generated_ids, history
(``steps`` full sequences, one after each denoising step), commits, refresh,
split_steps.
"""
from __future__ import annotations

import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from src.feature_cache_ext import StepObserver, feature_cache_session
from src.refresh_rules import make_policy
from src.utils import set_seed

LLADA_MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
LLADA_MASK_ID = 126336
LLADA_BLOCKS_PATH = "model.transformer.blocks"
LLADA_HOOK_MODULE = "dllm_cache.hooks.cache_hook_LLaDA"


def load_llada_model(model_name=LLADA_MODEL_NAME, device="cuda"):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to(device).eval()
    return model, tokenizer


def import_official_llada(repo):
    """(dLLMCache, dLLMCacheConfig, register_cache_LLaDA, logout_cache_LLaDA).

    ``repo`` is put on sys.path by exp9b.import_official(); call that first."""
    from dllm_cache.cache import dLLMCache, dLLMCacheConfig
    from dllm_cache.hooks import logout_cache_LLaDA, register_cache_LLaDA
    return dLLMCache, dLLMCacheConfig, register_cache_LLaDA, logout_cache_LLaDA


def llada_prepare_input(tokenizer, prompt, device):
    """Chat template exactly as in the official LLaDA / dLLM-Cache demos."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         add_generation_prompt=True, tokenize=False)
    enc = tokenizer(text)
    ids = torch.tensor(enc["input_ids"], device=device).unsqueeze(0)
    mask = torch.tensor(enc["attention_mask"], device=device).unsqueeze(0)
    return ids, mask


def llada_special_ids(tokenizer):
    """(mask id, ids that count as 'no content': EOS / EOT / pad)."""
    non_content = set()
    for value in (tokenizer.eos_token_id, tokenizer.pad_token_id):
        if isinstance(value, int):
            non_content.add(value)
    for tok in ("<|eot_id|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            non_content.add(tid)
    non_content.discard(LLADA_MASK_ID)
    return LLADA_MASK_ID, non_content


def add_gumbel_noise(logits, temperature):
    """Official LLaDA generate.py (float64 Gumbel-max)."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base, remainder = mask_num // steps, mask_num % steps
    out = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        out[i, :remainder[i]] += 1
    return out


def llada_generate(model, tokenizer, ids, mask, new_tokens, steps, seed, *,
                   alg="low_confidence", temperature=0.0, block_length=8,
                   cache_api=None, policy=None, prompt_tail_rows=0, force_prompt_tail=False,
                   capture_trajectory=True, log_confidence=True, observe=True,
                   link=None, link_cfg=None, device=None, **_ignored):
    """One LLaDA generation (no cache when cache_api is None).

    Same call signature as the Dream ``generate`` (``alg`` = LLaDA remasking:
    "low_confidence" or "random"; top_p / top_k / alg_temp are accepted and
    ignored). ``mask`` is ignored: a single unpadded sequence needs no attention
    mask, and LLaDA then builds no attention bias. ``device`` = where the sequence
    and the sampler live (the cloud GPU in a split; default ids.device).
    """
    if prompt_tail_rows:
        raise ValueError("LLaDA has no logit shift: prompt_tail_rows must be 0")
    if alg not in {"low_confidence", "random"}:
        raise ValueError(f"LLaDA remasking must be low_confidence or random, not {alg!r}")
    plen, gen_length, steps = int(ids.shape[1]), int(new_tokens), int(steps)
    block_length = int(block_length)
    if gen_length % block_length:
        raise ValueError(f"new_tokens={gen_length} must be a multiple of block_length={block_length}")
    num_blocks = gen_length // block_length
    if steps % num_blocks:
        raise ValueError(f"steps={steps} must be a multiple of the number of blocks ({num_blocks})")
    steps_per_block = steps // num_blocks
    device = torch.device(device) if device is not None else ids.device
    mask_id, non_content = llada_special_ids(tokenizer)
    if cache_api is not None and make_policy(policy).kind == "adaptive":
        observe = True

    set_seed(int(seed))
    session = (feature_cache_session(model, cache_api, plen, policy, 0, False,
                                     blocks_path=LLADA_BLOCKS_PATH,
                                     hook_module=LLADA_HOOK_MODULE)
               if cache_api is not None else nullcontext(None))
    with session as controller:
        observer = StepObserver(plen, mask_id, non_content, controller,
                                log_confidence and observe)
        if link is not None:
            link.begin(prompt_len=plen, controller=controller, prompt_tail_rows=0,
                       mask_id=mask_id, **(link_cfg or {}))
        set_seed(int(seed))
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        history = []
        x = torch.full((1, plen + gen_length), mask_id, dtype=torch.long, device=device)
        x[:, :plen] = ids.to(device)
        step = 0
        try:
            with torch.inference_mode():
                for b in range(num_blocks):
                    b_end = (b + 1) * block_length  # relative to the answer start
                    block_mask = x[:, plen + b * block_length:plen + b_end] == mask_id
                    n_transfer = get_num_transfer_tokens(block_mask, steps_per_block)
                    for i in range(steps_per_block):
                        mask_index = x[:, plen:] == mask_id
                        logits = model(x).logits
                        gen_logits = logits[:, plen:]
                        x0 = torch.argmax(add_gumbel_noise(gen_logits, temperature), dim=-1)
                        if alg == "low_confidence":
                            p = F.softmax(gen_logits, dim=-1)
                            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
                        else:
                            x0_p = torch.rand(x0.shape, device=x0.device)
                        x0_p[:, b_end:] = -np.inf
                        x0 = torch.where(mask_index, x0, x[:, plen:])
                        confidence = torch.where(mask_index, x0_p, -np.inf)
                        transfer = torch.zeros_like(x0, dtype=torch.bool)
                        for j in range(confidence.shape[0]):
                            sel = torch.topk(confidence[j], k=int(n_transfer[j, i])).indices
                            transfer[j, sel] = True
                        x[:, plen:][transfer] = x0[transfer]
                        if observe:
                            observer(step, x, logits)
                        if capture_trajectory:
                            history.append(x[0].detach().clone())
                        step += 1
        except BaseException:
            if link is not None:
                link.active = False
            raise
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        split_steps = link.end() if link is not None else None
        refresh = controller.summary() if controller is not None else None

    generated = x[0, plen:].detach().cpu().long().clone()
    if capture_trajectory:
        history = [h.cpu().long() for h in history]
        if len(history) != steps or not torch.equal(history[-1][plen:], generated):
            raise RuntimeError("LLaDA history does not match the final output")
    else:
        history = None
    return {"ms": elapsed_ms, "generated_ids": generated, "history": history,
            "commits": observer.commits, "refresh": refresh, "split_steps": split_steps}
