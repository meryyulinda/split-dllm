"""Exp13C: does splitting Fast-dLLM change its output?

Run from dasr1/ (two GPUs, timing not measured, so busy GPUs are fine):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 python experiments/exp13c_split_exactness.py

For each prompt the same Fast-dLLM configuration runs twice: first with the whole
model on the cloud GPU (no split, no link), then split after layer k with the
bf16 row rule (edge sends the block rows in-block and every row at block starts).
If the generated token sequences are identical on every prompt, the transmission
of the Fast-dLLM tier adds no error, as shown for dLLM-Cache in Exp11 (C2 vs C1).

Uses the Exp13 settings (EXP13_MODEL, EXP13_DATASET, EXP13_N_QUESTIONS,
EXP13_NEW_TOKENS, EXP13_BLOCK_LENGTH, EXP13_SPLIT_LAYER, EXP13_EDGE_DEVICE,
EXP13_CLOUD_DEVICE) plus
    EXP13C_VARIANTS   default ["K2"]; any of F0 (one token per step, no cache),
                      K1 (DualCache), K2 (DualCache + threshold 0.9)
Outputs: outputs/exp13c_exactness_<model>_<dataset>_k<split>_t<tokens>/
    runs.csv (case, variant, identical, first_diff, texts), summary.csv, metadata.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

import exp13_fastdllm_split as e13  # noqa: E402  (reads the EXP13_* settings)
from exp10_common import ROOT, describe  # noqa: E402
from src.fastdllm_split import KVBlockLink, kv_generate, load_fastdllm_model  # noqa: E402
from src.split_runtime import place_split  # noqa: E402

SPECS = {"F0": ("none", "fixed", None), "K1": ("dual", "fixed", None),
         "K2": ("dual", "threshold", e13.g0)}
NAMES = list(e13.setting("EXP13C_VARIANTS", ["K2"]))
OUT = ROOT / "outputs" / (f"exp13c_exactness_{e13.MODEL}_{e13.DATASET}_k{e13.SPLIT}"
                          f"_t{e13.NEW_TOKENS}")


def generate(model, tokenizer, ids, spec, link, device):
    cache, decode, thr = spec
    return kv_generate(model, e13.MODEL, tokenizer, ids, e13.NEW_TOKENS, e13.BLOCK, e13.SEED,
                       cache=cache, decode=decode, threshold=thr,
                       temperature=e13.TEMPERATURE, top_p=e13.TOP_P, link=link,
                       link_cfg={"encoding": "bf16", "anchor_interval": e13.ANCHOR}
                       if link is not None else None,
                       device=device)["generated_ids"]


def main():
    unknown = [n for n in NAMES if n not in SPECS]
    if unknown:
        raise ValueError(f"unknown EXP13C_VARIANTS {unknown}; known: {list(SPECS)}")
    OUT.mkdir(parents=True, exist_ok=True)
    cases, _ = e13.load_cases()
    cloud = torch.device(e13.CLOUD)
    model, tokenizer = load_fastdllm_model(e13.MODEL, e13.MODEL_NAME, e13.REPO, device=cloud)
    print(f"Exp13C | {e13.MODEL} | {e13.DATASET} n={len(cases)} | variants {NAMES}", flush=True)

    single = {}
    for case in cases:                      # 1) whole model on the cloud GPU
        ids, _ = e13.prepare(tokenizer, case["prompt"], cloud)
        for n in NAMES:
            single[(case["case_id"], n)] = generate(model, tokenizer, ids, SPECS[n], None, cloud)
    print("single-GPU runs done", flush=True)

    edge, cloud = place_split(model, e13.SPLIT, e13.EDGE, e13.CLOUD)   # 2) split
    link = KVBlockLink(model, e13.SPLIT, edge, cloud)
    rows = []
    for case in cases:
        ids, _ = e13.prepare(tokenizer, case["prompt"], cloud)
        for n in NAMES:
            split = generate(model, tokenizer, ids, SPECS[n], link, cloud)
            ref = single[(case["case_id"], n)]
            same = bool(torch.equal(ref, split))
            diff = None if same else int((ref != split).nonzero()[0, 0])
            rows.append({"case_id": case["case_id"], "variant": n, "identical": same,
                         "first_diff_position": diff,
                         "text_single": describe(tokenizer, ref)["text"],
                         "text_split": describe(tokenizer, split)["text"]})
            print(f"  {case['case_id']} {n}: {'identical' if same else f'differs at {diff}'}",
                  flush=True)
    link.remove()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "runs.csv", index=False)
    summary = df.groupby("variant")["identical"].agg(["sum", "size"]).reset_index()
    summary.columns = ["variant", "identical", "prompts"]
    summary.to_csv(OUT / "summary.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps({
        "experiment": "Exp13C split exactness of Fast-dLLM", "model": e13.MODEL_NAME,
        "dataset": e13.DATASET, "n_cases": len(cases), "split_layer_1based": e13.SPLIT,
        "new_tokens": e13.NEW_TOKENS, "block_length": e13.BLOCK, "variants": NAMES,
        "temperature": e13.TEMPERATURE, "top_p": e13.TOP_P, "encoding": "bf16",
    }, indent=2), encoding="utf-8")
    print("\n" + summary.to_string(index=False))
    print("Exp13C completed. Output:", OUT)


if __name__ == "__main__":
    main()
