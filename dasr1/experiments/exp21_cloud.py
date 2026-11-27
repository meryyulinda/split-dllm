"""Exp21 cloud program: serves edge devices over TCP (see src/remote_split.py).

Run from dasr1/ on the lab server, one process per (backend, split layer):
  CUDA_VISIBLE_DEVICES=1 python experiments/exp21_cloud.py --backend fastdllm --split 4 --port 6003
--split 0 = cloud-only: the cloud runs every layer and the edge only sends the prompt and
receives the answer (the cloud-only reference over the same network path).

It listens on 127.0.0.1 only. An edge on another machine reaches it through an SSH
tunnel (see run_exp21_cloud.sh). Each connection is served until the edge sends BYE;
connections are served one after another (one answer at a time).
Every answer's per-step statistics are appended to outputs/exp21_cloud_logs/<name>.jsonl.
"""
import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from exp10_common import ROOT, cfg  # noqa: E402
from src.remote_split import (F_BEGIN, F_BYE, F_DONE, RemoteLink, _tune, recv_frame,  # noqa: E402
                              send_frame, strip_edge)


def load(backend, device):
    if backend == "fastdllm":
        from src.fastdllm_split import DREAM_MODEL_NAME, load_fastdllm_model
        model, tok = load_fastdllm_model("dream", getattr(cfg, "MODEL_NAME", DREAM_MODEL_NAME),
                                         getattr(cfg, "FASTDLLM_REPO", "../../Fast-dLLM"),
                                         device=device)
        return model, tok, None
    from exp9b_official_feature_cache_diagnostics import import_official
    from src.model_loader import load_dream_model
    _, api = import_official()
    model, tok = load_dream_model(cfg.MODEL_NAME)
    return model.to(device).eval(), tok, api


def run_session(c, model, tok, api, link, device, backend, conn):
    ids = torch.tensor([c["prompt_ids"]], dtype=torch.long, device=device)
    plen, n_tok = int(ids.shape[1]), int(c["new_tokens"])
    if link is not None:
        link.sock = conn
    t0 = time.perf_counter()
    if backend == "fastdllm":
        from src.fastdllm_split import kv_generate
        res = kv_generate(model, "dream", tok, ids, n_tok, int(c["block_length"]), int(c["seed"]),
                          cache=c["cache"], decode=c["decode"], threshold=c.get("threshold"),
                          temperature=0.0, top_p=None, link=link,
                          link_cfg={"encoding": c["encoding"], "anchor_interval": c["anchor_interval"],
                                    "cache": c["cache"]} if link is not None else None,
                          device=device)
    else:
        from src.feature_cache_ext import generate as dream_generate
        policy = tuple(c["policy"]) if c.get("policy") else None
        mask = torch.ones_like(ids)
        res = dream_generate(model, tok, ids, mask, n_tok, int(c["steps"]), int(c["seed"]),
                             alg=c["alg"], temperature=float(c["temperature"]),
                             top_p=c.get("top_p"), cache_api=api if policy else None,
                             policy=policy, prompt_tail_rows=int(c["prompt_tail_rows"]) if policy else 0,
                             capture_trajectory=False, log_confidence=False, link=link,
                             link_cfg={"rows": c["rows"], "encoding": c["encoding"],
                                       "anchor_interval": c["anchor_interval"]}
                             if link is not None else None)
    gen_s = time.perf_counter() - t0
    gen = res["generated_ids"].to(device)
    with torch.inference_mode():
        full = torch.cat([ids, gen.unsqueeze(0)], 1)
        if link is not None and link.record is not None:
            idx = torch.nonzero(full[0] != link.record[0]).squeeze(1)
        else:                                   # cloud-only: the edge holds only the prompt
            idx = torch.arange(plen, full.shape[1], device=device)
        final_pos, final_tok = idx.tolist(), full[0, idx].tolist()
    steps = res.get("split_steps") or []
    info = {"final_pos": final_pos, "final_tok": final_tok, "cloud_gen_s": gen_s,
            "n_steps": len(steps) or int(res.get("n_steps") or c.get("steps", 0)),
            "cloud_ms": sum(s.get("cloud_ms", 0) for s in steps),
            "sample_ms": sum(s.get("sample_ms", 0) for s in steps),
            "wire_ms": sum(s.get("wire_ms", 0) for s in steps),
            "edge_ms_reported": sum(s.get("edge_ms", 0) for s in steps),
            "up_bytes": sum(s.get("up_bytes", 0) for s in steps),
            "down_bytes": sum(s.get("down_bytes", 0) for s in steps)}
    return info, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["dllm_cache", "fastdllm"], required=True)
    ap.add_argument("--split", type=int, default=4)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    device = torch.device(a.device)
    torch.cuda.set_device(device)
    name = f"{a.backend}_k{a.split}"
    model, tok, api = load(a.backend, device)
    link = None
    if a.split > 0:
        link = RemoteLink(model, device, model.config.hidden_size)
        strip_edge(model, a.split, link)
    logdir = ROOT / "outputs" / "exp21_cloud_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.host, a.port))
    srv.listen(4)
    print(f"Exp21 cloud | {name} | {torch.cuda.get_device_name(device)} | listening on "
          f"{a.host}:{a.port}", flush=True)
    while True:
        conn, peer = srv.accept()
        _tune(conn)
        print(f"{time.strftime('%F %T')} edge connected from {peer}", flush=True)
        try:
            while True:
                ftype, body, _ = recv_frame(conn)
                if ftype == F_BYE:
                    break
                if ftype != F_BEGIN:
                    raise RuntimeError(f"cloud: unexpected frame type {ftype}")
                c = json.loads(bytes(body).decode())
                info, steps = run_session(c, model, tok, api, link, device, a.backend, conn)
                send_frame(conn, F_DONE, json.dumps(info).encode())
                with (logdir / f"{name}.jsonl").open("a") as f:
                    f.write(json.dumps({"time": time.strftime("%F %T"), "case_id": c.get("case_id"),
                                        "encoding": c.get("encoding"), "gpu": torch.cuda.get_device_name(device),
                                        **{k: v for k, v in info.items() if not k.startswith("final")},
                                        "steps": steps}) + "\n")
                print(f"  {c.get('case_id')} {c.get('encoding')}: {info['n_steps']} steps, "
                      f"cloud {info['cloud_gen_s']:.1f} s", flush=True)
        except (EOFError, ConnectionError, OSError) as exc:
            print(f"connection ended: {exc!r}", flush=True)
        finally:
            conn.close()
            if link is not None:
                link.active = False


if __name__ == "__main__":
    main()
