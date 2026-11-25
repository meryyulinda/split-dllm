"""Exp15: how much of the prompt can be read back from the boundary activations?

Run from dasr1/ on ONE GPU (timing is not measured, so a busy GPU is fine):
    CUDA_VISIBLE_DEVICES=0 python experiments/exp15_boundary_leakage.py

In a split after layer k, the edge sends the input of layer k+1 (the output of
layer k) for every prompt row at least once. This experiment measures, for
several k, how many prompt tokens an observer of the link could recover from
those rows, with three attacks of increasing strength:

  embed_nn   no training: nearest input-embedding row (cosine) to the boundary
             row. Works when the boundary still looks like the embedding.
  lm_lens    no training: apply the model's own final norm and LM head to the
             boundary row ("logit lens"); count a hit if the top token equals
             the token at the same position or at the next one (Dream predicts
             position i+1 from row i). Needs only the public weights.
  ridge      trained: a linear map from boundary rows to input embeddings,
             fitted by ridge regression on hidden states of training prompts,
             decoded by nearest embedding. An attacker with the public model can
             produce such training data offline.

The boundary is captured exactly as the split transmits it: a forward pre-hook
on the first cloud layer, on the first denoising step (prompt followed by
masked answer positions), which every split sends in full.

Data: inverter training on GSM8K *train* questions (Hugging Face `openai/gsm8k`,
falling back to local test questions 100..199 if the Hub is unreachable);
evaluation on held-out prompts: GSM8K test questions 0..99 and the 12 authored
long-prompt cases (documents). Metrics are computed on the private part of each
prompt only: the GSM8K question, or the document and question of an authored case.
Chat-template tokens and fixed instructions are excluded, since they are the same
in every request and reveal nothing.

Outputs: outputs/exp15_leakage_<model>/
    summary.csv      model, layer, attack, test_set, top1, top5, top1_digits, n_rows
    examples.md      reconstructed prompt text (best attack) at a few layers
    metadata.json
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

import configs.experiment_config as cfg  # noqa: E402
from src.split_runtime import model_parts  # noqa: E402


def setting(name, default):
    return getattr(cfg, name, default)


MODEL = str(setting("EXP15_MODEL", "dream")).lower()
DEFAULT_LAYERS = {"dream": [1, 2, 4, 8, 12, 16, 20, 24], "llada": [1, 2, 4, 8, 12, 16, 20, 24, 28]}
LAYERS = [int(k) for k in setting("EXP15_LAYERS", DEFAULT_LAYERS.get(MODEL, [1, 2, 4, 8]))]
N_TRAIN = int(setting("EXP15_N_TRAIN", 2000))          # training prompts for the ridge inverter
N_VAL = int(setting("EXP15_N_VAL", 200))               # of those, held out to pick lambda
N_TEST_GSM8K = int(setting("EXP15_N_TEST_GSM8K", 100))
GEN_MASKS = int(setting("EXP15_GEN_MASKS", 64))         # masked answer positions after the prompt
LAMBDAS = [float(x) for x in setting("EXP15_LAMBDAS", [1e-3, 1e-2, 1e-1])]
DEVICE = setting("EXP15_DEVICE", "cuda:0")
OUTPUT_TAG = str(setting("EXP15_OUTPUT_TAG", ""))
OUT = ROOT / "outputs" / f"exp15_leakage_{MODEL}{OUTPUT_TAG}"
EXAMPLE_LAYERS = [k for k in (1, 4, 8, 16) if k in LAYERS]


# --------------------------------------------------------------------------- model
def load_model():
    if MODEL == "dream":
        from src.model_loader import load_dream_model
        from exp9b_official_feature_cache_diagnostics import prepare_input
        model, tok = load_dream_model(setting("MODEL_NAME", "Dream-org/Dream-v0-Instruct-7B"))
        mask_id = tok.mask_token_id

        def prep(prompt):
            ids, _ = prepare_input(tok, prompt, DEVICE)
            return ids
    elif MODEL == "llada":
        from src.llada_cache_ext import LLADA_MASK_ID, llada_prepare_input, load_llada_model
        model, tok = load_llada_model(setting("LLADA_MODEL_NAME", "GSAI-ML/LLaDA-8B-Instruct"),
                                      device=DEVICE)
        mask_id = LLADA_MASK_ID

        def prep(prompt):
            ids, _ = llada_prepare_input(tok, prompt, DEVICE)
            return ids
    else:
        raise ValueError(f"EXP15_MODEL must be 'dream' or 'llada', not {MODEL!r}")
    model = model.to(DEVICE).eval()
    return model, tok, int(mask_id), prep


FALLBACKS = {"n": 0}


def content_mask(tok, ids, private_text):
    """True at the positions of the private text inside the chat-templated prompt."""
    seq = ids[0].tolist()
    mask = torch.zeros(len(seq), dtype=torch.bool)
    target = tok(private_text, add_special_tokens=False)["input_ids"]
    for cut in (0, 1, 2):                      # boundary tokens may merge with the template
        t = target[cut:len(target) - cut] if cut else target
        if len(t) < 4:
            break
        for i in range(len(seq) - len(t) + 1):
            if seq[i:i + len(t)] == t:
                mask[i:i + len(t)] = True
                return mask
    FALLBACKS["n"] += 1                      # not found: every non-special token
    special = set(tok.all_special_ids)
    return torch.tensor([s not in special for s in seq])


# --------------------------------------------------------------------------- data
def train_prompts():
    from exp10d_gsm8k_refresh import INSTRUCTION
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        qs = [r["question"] for r in ds.select(range(min(N_TRAIN, len(ds))))]
        source = "openai/gsm8k train"
    except Exception as err:  # Hub unreachable: local test questions 100..199
        print(f"[exp15] GSM8K train split unavailable ({err!r}); using local test 100..199",
              flush=True)
        from exp10d_gsm8k_refresh import load_cases
        qs = [c["question"] for c in load_cases(n=100, start=100)]
        source = "local gsm8k test 100..199 (fallback)"
    return [(q.strip() + "\n" + INSTRUCTION, q.strip()) for q in qs], source


def test_sets():
    from exp10d_gsm8k_refresh import load_cases
    from exp10a_official_feature_cache_quality import make_cases
    return {"gsm8k_test": [(c["prompt"], c["question"].strip())
                           for c in load_cases(n=N_TEST_GSM8K, start=0)],
            # private part: the document and the question (the fixed preface is public)
            "authored_docs": [(c["prompt"], c["prompt"].split("\n\n", 1)[1])
                              for c in make_cases()]}


# --------------------------------------------------------------------------- capture
class Boundary:
    """Forward pre-hooks on layer k (0-based) capture the input of layer k+1, i.e. the
    boundary of a split after k layers."""

    def __init__(self, layers, ks):
        self.store = {}
        self.handles = [layers[k].register_forward_pre_hook(self._hook(k), with_kwargs=True)
                        for k in ks]

    def _hook(self, k):
        def fn(module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            self.store[k] = h[0].detach()
            return None
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def boundary_rows(model, cap, prep, tok, item, mask_id):
    prompt, private_text = item
    ids = prep(prompt)
    lp = ids.shape[1]
    x = torch.cat([ids, torch.full((1, GEN_MASKS), mask_id, dtype=ids.dtype, device=ids.device)], 1)
    cap.store.clear()
    model(x)
    keep = content_mask(tok, ids, private_text).to(ids.device)
    tokens = ids[0]
    nxt = torch.cat([tokens[1:], x[0, lp:lp + 1]])           # token at position i+1
    return {k: v[:lp][keep] for k, v in cap.store.items()}, tokens[keep], nxt[keep]


# --------------------------------------------------------------------------- attacks
class RidgeStats:
    def __init__(self, d, device):
        self.n = 0
        self.sh = torch.zeros(d, device=device, dtype=torch.float64)
        self.se = torch.zeros(d, device=device, dtype=torch.float64)
        self.hh = torch.zeros(d, d, device=device, dtype=torch.float32)
        self.he = torch.zeros(d, d, device=device, dtype=torch.float32)

    def add(self, h, e):
        h, e = h.float(), e.float()
        self.n += h.shape[0]
        self.sh += h.sum(0).double()
        self.se += e.sum(0).double()
        self.hh += h.T @ h
        self.he += h.T @ e

    def solve(self, lam):
        n = float(self.n)
        mh, me = (self.sh / n).float(), (self.se / n).float()
        chh = self.hh / n - torch.outer(mh, mh)
        che = self.he / n - torch.outer(mh, me)
        s = chh.diagonal().clamp_min(1e-8).sqrt()
        c = chh / torch.outer(s, s)
        c.diagonal().add_(lam)
        w = torch.linalg.solve(c.double(), (che / s[:, None]).double()).float()
        return {"mh": mh, "me": me, "s": s, "w": w}

    @staticmethod
    def apply(fit, h):
        return ((h.float() - fit["mh"]) / fit["s"]) @ fit["w"] + fit["me"]


def nearest(vecs, emb_n, k=5, chunk=1024):
    out = []
    for i in range(0, vecs.shape[0], chunk):
        v = torch.nn.functional.normalize(vecs[i:i + chunk].float(), dim=-1).to(emb_n.dtype)
        out.append((v @ emb_n.T).topk(k, dim=-1).indices)
    return torch.cat(out) if out else torch.zeros(0, k, dtype=torch.long, device=emb_n.device)


def lens_topk(parts, h, k=5, chunk=512):
    out = []
    for i in range(0, h.shape[0], chunk):
        z = h[i:i + chunk].unsqueeze(0)
        for m in parts.cloud_extra:      # final norm
            z = m(z)
        out.append(parts.head(z)[0].float().topk(k, dim=-1).indices)
    return torch.cat(out)


_DIGIT = {}


def _has_digit(tok, t):
    if t not in _DIGIT:
        _DIGIT[t] = any(ch.isdigit() for ch in tok.decode([t]))
    return _DIGIT[t]


def score(pred5, tokens, tok, nxt=None):
    hit1 = pred5[:, 0] == tokens
    hit5 = (pred5 == tokens[:, None]).any(1)
    if nxt is not None:
        hit1 = hit1 | (pred5[:, 0] == nxt)
        hit5 = hit5 | (pred5 == nxt[:, None]).any(1)
    digits = torch.tensor([_has_digit(tok, t) for t in tokens.tolist()], device=tokens.device)
    d1 = float(hit1[digits].float().mean()) if digits.any() else float("nan")
    return float(hit1.float().mean()), float(hit5.float().mean()), d1


# --------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok, mask_id, prep = load_model()
    parts = model_parts(model)
    n_layers = len(parts.layers)
    ks = [k for k in LAYERS if 1 <= k < n_layers]
    emb = parts.embed.weight.detach()
    emb_n = torch.nn.functional.normalize(emb.float(), dim=-1).to(torch.bfloat16)
    cap = Boundary(parts.layers, ks)
    d = emb.shape[1]
    print(f"Exp15 | {MODEL} | {n_layers} layers | boundaries after layers {ks} | d={d}", flush=True)

    prompts, train_source = train_prompts()
    n_val = min(N_VAL, max(len(prompts) // 10, 1))
    fit_prompts, val_prompts = prompts[:-n_val], prompts[-n_val:]
    stats = {k: RidgeStats(d, DEVICE) for k in ks}
    for j, p in enumerate(fit_prompts):
        rows, toks, _ = boundary_rows(model, cap, prep, tok, p, mask_id)
        e = emb[toks]
        for k in ks:
            stats[k].add(rows[k], e)
        if (j + 1) % 200 == 0:
            print(f"  fit prompts {j + 1}/{len(fit_prompts)}", flush=True)

    def collect(prompt_list):
        acc = {k: [] for k in ks}
        toks_all, nxt_all, per_prompt = [], [], []
        for p in prompt_list:
            rows, toks, nxt = boundary_rows(model, cap, prep, tok, p, mask_id)
            for k in ks:
                acc[k].append(rows[k])
            toks_all.append(toks)
            nxt_all.append(nxt)
            per_prompt.append(len(toks))
        return ({k: torch.cat(v) for k, v in acc.items()}, torch.cat(toks_all),
                torch.cat(nxt_all), per_prompt)

    val_rows, val_tok, _, _ = collect(val_prompts)
    fits, chosen = {}, {}
    for k in ks:
        best = None
        for lam in LAMBDAS:
            f = stats[k].solve(lam)
            acc = score(nearest(RidgeStats.apply(f, val_rows[k]), emb_n), val_tok, tok)[0]
            if best is None or acc > best[0]:
                best = (acc, lam, f)
        chosen[k], fits[k] = best[1], best[2]
        print(f"  layer {k}: lambda={best[1]:g} (val top1 {best[0]:.3f})", flush=True)
        stats[k] = None
    torch.cuda.empty_cache()

    shift = MODEL == "dream"
    results, examples = [], []
    for name, plist in test_sets().items():
        rows, toks, nxt, per_prompt = collect(plist)
        for k in ks:
            preds = {
                "embed_nn": nearest(rows[k], emb_n),
                "lm_lens": lens_topk(parts, rows[k]),
                "ridge": nearest(RidgeStats.apply(fits[k], rows[k]), emb_n),
            }
            for attack, p5 in preds.items():
                t1, t5, dig = score(p5, toks, tok, nxt if attack == "lm_lens" else None)
                results.append({"model": MODEL, "layer": k, "attack": attack, "test_set": name,
                                "top1": round(t1, 4), "top5": round(t5, 4),
                                "top1_digits": round(dig, 4), "n_rows": int(toks.numel())})
            if k in EXAMPLE_LAYERS:
                best = max(("embed_nn", "lm_lens", "ridge"),
                           key=lambda a: next(r["top1"] for r in results
                                              if r["layer"] == k and r["attack"] == a
                                              and r["test_set"] == name))
                n0 = per_prompt[0]
                examples.append((name, k, best, tok.decode(toks[:n0].tolist()),
                                 tok.decode(preds[best][:n0, 0].tolist())))
        print(f"  {name}: {toks.numel()} content tokens evaluated", flush=True)
    cap.remove()

    df = pd.DataFrame(results)
    df.to_csv(OUT / "summary.csv", index=False)
    with open(OUT / "examples.md", "w", encoding="utf-8") as f:
        f.write(f"# Exp15 reconstructions ({MODEL}, first prompt of each test set)\n\n")
        for name, k, attack, orig, rec in examples:
            f.write(f"## {name}, boundary after layer {k} ({attack})\n\n"
                    f"**Original:**\n\n{orig}\n\n**Recovered (top-1 per row):**\n\n{rec}\n\n")
    (OUT / "metadata.json").write_text(json.dumps({
        "experiment": "Exp15 boundary leakage", "model": MODEL, "layers": ks,
        "train_source": train_source, "n_fit_prompts": len(fit_prompts),
        "n_val_prompts": len(val_prompts), "ridge_lambda": chosen, "gen_masks": GEN_MASKS,
        "logit_shift_checked": shift, "private_text_not_found": FALLBACKS["n"],
        "seconds": round(time.time() - t0, 1),
        "gpu": torch.cuda.get_device_name(DEVICE), "torch": torch.__version__,
        "transformers": transformers.__version__, "python": platform.python_version(),
    }, indent=2), encoding="utf-8")
    pivot = df.pivot_table(index=["test_set", "attack"], columns="layer", values="top1")
    print("\nTop-1 token recovery (content tokens):")
    print(pivot.round(3).to_string())
    print("\nExp15 completed. Output:", OUT)


if __name__ == "__main__":
    main()
