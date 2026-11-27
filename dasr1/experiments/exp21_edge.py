"""Exp21 edge program: DASR's edge part on its own machine, talking to the cloud program.

Run from dasr1/ on the edge machine (a rented RTX 3060, or one GPU of the lab server):
  CUDA_VISIBLE_DEVICES=0 python experiments/exp21_edge.py          (settings: EXP21_*)

For every cloud server in EXP21_SERVERS (backend, split layer, address) and every question,
it sends the prompt, computes the edge part of every denoising step the cloud asks for,
and measures the real time per answer from sending the prompt until it holds the answer.
Split servers run each codec of EXP21_CODECS; split-0 servers are the cloud-only reference.

Checks: answers vs the single-process runs of Exp16 ("local", same settings), which must be
identical when the edge runs on the same GPU type (A6000) with bf16 rows; on another GPU
the edge's arithmetic differs slightly, so some answers may change.

Outputs: outputs/exp21_deploy_<EXP21_TAG>/ runs.csv, summary.csv, metadata.json
"""
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from exp10_common import ROOT, cfg, describe  # noqa: E402
from src.remote_split import (F_BYE, EdgeRunner, connect, import_dllm_cache,  # noqa: E402
                              load_edge_model, send_frame)


def setting(name, default):
    return getattr(cfg, name, default)


TAG = str(setting("EXP21_TAG", "lab"))
DEVICE = setting("EXP21_EDGE_DEVICE", "cuda:0")
SPLIT = int(setting("EXP21_SPLIT_LAYER", 4))
SERVERS = list(setting("EXP21_SERVERS", [
    {"name": "dllm_cache_split", "backend": "dllm_cache", "split": SPLIT, "addr": "127.0.0.1:6001"},
    {"name": "dllm_cache_cloud_only", "backend": "dllm_cache", "split": 0, "addr": "127.0.0.1:6002"},
    {"name": "fastdllm_split", "backend": "fastdllm", "split": SPLIT, "addr": "127.0.0.1:6003"},
    {"name": "fastdllm_cloud_only", "backend": "fastdllm", "split": 0, "addr": "127.0.0.1:6004"},
]))
CODECS = list(setting("EXP21_CODECS", ["bf16", "int8_delta"]))
DATASET = setting("EXP21_DATASET", "gsm8k")
N = int(setting("EXP21_N_QUESTIONS", 10))
NEW_TOKENS = int(setting("EXP21_NEW_TOKENS", 256))
SLOWDOWN = float(setting("EXP21_EDGE_SLOWDOWN", 1.0))     # 1 on a real edge GPU
SEED = int(setting("SEED", 42))
POLICY = list(setting("EXP11_CACHE_POLICY", ("fixed", 25, 2, 0.25)))
DLLM_REPO = setting("EXP21_DLLM_CACHE_REPO", setting("EXP9A_OFFICIAL_CACHE_REPO",
                                                      "../../dLLM-cache-official"))
FAST_REPO = setting("FASTDLLM_REPO", "../../Fast-dLLM")
MODEL_NAME = setting("MODEL_NAME", "Dream-org/Dream-v0-Instruct-7B")
REFERENCE = {   # single-process runs with the same settings (Exp16, link "local")
    ("dllm_cache", "bf16"): ("outputs/exp16_dllm_cache_dream_gsm8k_k4_t256/runs.csv", "C2_cache_aware"),
    ("dllm_cache", "int8_delta"): ("outputs/exp16_dllm_cache_dream_gsm8k_k4_t256/runs.csv", "C4_cache_aware_int8"),
    ("fastdllm", "bf16"): ("outputs/exp16_fastdllm_dream_gsm8k_k4_t256/runs.csv", "K2_dual_thr0.9"),
    ("fastdllm", "int8_delta"): ("outputs/exp16_fastdllm_dream_gsm8k_k4_t256/runs.csv", "K3_dual_thr0.9_int8"),
}
PAPER = {("dllm_cache", "bf16"): "DASR + dLLM-Cache (bf16)",
         ("dllm_cache", "int8_delta"): "DASR + dLLM-Cache (INT8)",
         ("fastdllm", "bf16"): "DASR + Fast-dLLM (bf16)",
         ("fastdllm", "int8_delta"): "DASR + Fast-dLLM (INT8)",
         ("dllm_cache", "cloud_only"): "Cloud-only + dLLM-Cache",
         ("fastdllm", "cloud_only"): "Cloud-only + Fast-dLLM"}
OUT = ROOT / "outputs" / f"exp21_deploy_{TAG}"


def session_cfg(backend, split, encoding, case_id):
    c = {"backend": backend, "split": split, "encoding": encoding, "anchor_interval": 32,
         "new_tokens": NEW_TOKENS, "seed": SEED, "case_id": case_id}
    if backend == "fastdllm":
        c.update(cache="dual", decode="threshold", threshold=0.9, block_length=32)
    else:
        c.update(steps=NEW_TOKENS, alg="entropy", temperature=0.1, top_p=0.9, policy=POLICY,
                 prompt_tail_rows=1, rows="cache_aware", transfer_ratio=float(POLICY[3]))
    return c


def load_reference():
    ref = {}
    for key, (path, variant) in REFERENCE.items():
        p = ROOT / path
        if p.is_file():
            df = pd.read_csv(p)
            df = df[(df["link"] == "local") & (df["variant"] == variant)]
            ref[key] = dict(zip(df["case_id"], df["text"]))
    return ref


def main():
    if DATASET == "gsm8k":
        from exp10d_gsm8k_refresh import gsm8k_score, load_cases
        cases, score_fn = load_cases(n=N, start=0), gsm8k_score
    else:
        from exp10_common import default_score
        from exp10a_official_feature_cache_quality import make_cases
        cases, score_fn = make_cases()[:N], default_score
    from exp9b_official_feature_cache_diagnostics import prepare_input
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mask_id = int(tok.mask_token_id)
    dev = torch.device(DEVICE)
    need = {s["backend"] for s in SERVERS if int(s["split"]) > 0}
    models, api = {}, None
    if "dllm_cache" in need:
        root = Path(DLLM_REPO)
        api = import_dllm_cache(root if root.is_absolute() else (ROOT / root))
        models["dllm_cache"] = load_edge_model("dllm_cache", MODEL_NAME, SPLIT, dev)
    if "fastdllm" in need:
        repo = Path(FAST_REPO)
        models["fastdllm"] = load_edge_model("fastdllm", MODEL_NAME, SPLIT, dev,
                                             repo if repo.is_absolute() else (ROOT / repo))
    runner = EdgeRunner(models, dev, api=api, slowdown=SLOWDOWN)
    ref = load_reference()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Exp21 edge | {torch.cuda.get_device_name(dev)} | split after layer {SPLIT} | "
          f"edge slowdown {SLOWDOWN:g} | servers {[s['name'] for s in SERVERS]}", flush=True)
    runs = []
    for srv in SERVERS:
        backend, split = srv["backend"], int(srv["split"])
        encodings = CODECS if split > 0 else ["cloud_only"]
        print(f"\n=== {srv['name']} at {srv['addr']}  ({time.strftime('%F %T')})", flush=True)
        sock = connect(srv["addr"])
        try:
            warm = prepare_input(tok, cases[0]["prompt"], "cpu")[0][0].tolist()
            runner.session(sock, session_cfg(backend, split, encodings[0] if split else "bf16",
                                             "warmup"), warm, mask_id)
            for case in cases:
                ids = prepare_input(tok, case["prompt"], "cpu")[0][0].tolist()
                for enc in encodings:
                    c = session_cfg(backend, split, enc if split else "bf16", case["case_id"])
                    answer, info = runner.session(sock, c, ids, mask_id)
                    text = describe(tok, answer)["text"]
                    score = score_fn(case, text)
                    r = ref.get((backend, enc if split else "bf16"), {})
                    row = {"server": srv["name"], "backend": backend, "split": split,
                           "codec": enc, "paper_name": PAPER[(backend, enc)],
                           "case_id": case["case_id"], "prompt_tokens": len(ids),
                           "latency_s": info["edge_wall_s"], "ok": bool(score["ok"]),
                           "extracted": score.get("extracted"),
                           "same_as_single_process": (text == r[case["case_id"]])
                           if case["case_id"] in r else "", "text": text,
                           **{k: v for k, v in info.items()}}
                    runs.append(row)
                    pd.DataFrame(runs).to_csv(OUT / "runs.csv", index=False)
                    print(f"  {case['case_id']} {enc:<11s} ok={row['ok']!s:<5} "
                          f"same={row['same_as_single_process']!s:<5} steps={info['n_steps']:4d} "
                          f"time={info['edge_wall_s']:7.2f} s (edge {info['edge_compute_ms'] / 1e3:5.2f} s, "
                          f"cloud {info['cloud_ms'] / 1e3:5.2f} s, wire {info['wire_ms'] / 1e3:5.2f} s)",
                          flush=True)
        finally:
            try:
                send_frame(sock, F_BYE)
            finally:
                sock.close()
    df = pd.DataFrame(runs)
    df["same"] = df["same_as_single_process"].map(lambda v: v is True)
    summary = df.groupby(["server", "codec"], sort=False).agg(
        paper_name=("paper_name", "first"), runs=("case_id", "size"), acc=("ok", "mean"),
        same_as_single_process=("same", "sum"), latency_s=("latency_s", "mean"),
        steps=("n_steps", "mean"), up_mb=("edge_up_bytes", lambda x: x.mean() / 1e6),
        edge_s=("edge_compute_ms", lambda x: x.mean() / 1e3),
        cloud_s=("cloud_ms", lambda x: x.mean() / 1e3),
        cloud_gen_s=("cloud_gen_s", "mean"),
        wire_s=("wire_ms", lambda x: x.mean() / 1e3)).reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps({
        "experiment": "Exp21 DASR as two programs (edge and cloud) over a real network",
        "tag": TAG, "edge_gpu": torch.cuda.get_device_name(dev), "servers": SERVERS,
        "codecs": CODECS, "dataset": DATASET, "n_cases": len(cases), "new_tokens": NEW_TOKENS,
        "split_layer_1based": SPLIT, "edge_slowdown": SLOWDOWN, "seed": SEED,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "host": platform.node(),
    }, indent=2, default=str), encoding="utf-8")
    print("\nExp21 edge completed. Output:", OUT)
    print(summary.drop(columns=["paper_name"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
