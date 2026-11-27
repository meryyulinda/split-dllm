"""Exp22 edge program: N users at once against one cloud program (exp22_cloud.py).

Question: how many users can one cloud GPU serve, and how fast is each answer, when
the model is split (the cloud runs layers k+1..L) versus cloud-only (the cloud runs
every layer)? Fast-dLLM backend, bf16 rows, Dream, GSM8K.

Each user is a thread with its own TCP connection, its own emulated access link (the
relay of src/net_relay.py --duplex gives every connection its own link) and its own
CUDA stream; the users share one copy of the edge layers on the edge GPU, whose steps
are slowed by --slowdown (time.sleep, as in Exp21) to stand for a slower edge GPU.
All N users start together (barrier) and each answers Q = max(pool // N, min_q)
questions from a pool of the first `pool` GSM8K problems; user u starts at question
u * pool // N, so the users spread over the pool.

Measured per answer: time from sending the prompt until the edge holds the answer,
steps, edge / cloud / network time, and whether the answer equals the same question's
answer with one user (concurrency must not change answers).
Per (configuration, link, users):
  answers_per_hour   each user's answers divided by that user's active time, summed
                     over the users (users ask again as soon as they hold an answer):
                     the throughput of the one cloud GPU
  latency_s          mean, median and 90th percentile time per answer

Output: outputs/exp22_multiuser_<tag>/ runs.csv, summary.csv, metadata.json.
A restarted run skips the (configuration, link, users) combinations already in runs.csv.
"""
import argparse
import json
import platform
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from exp10_common import ROOT, cfg, describe  # noqa: E402
from exp21_edge import session_cfg  # noqa: E402
from src.remote_split import F_BYE, EdgeRunner, connect, load_edge_model, send_frame  # noqa: E402

MODEL_NAME = getattr(cfg, "MODEL_NAME", "Dream-org/Dream-v0-Instruct-7B")
FAST_REPO = getattr(cfg, "FASTDLLM_REPO", "../../Fast-dLLM")
KEYS = ["config", "link", "users"]


def parse_links(text):
    """name=host:port:mbps:rtt_ms,... -> list of dicts"""
    out = []
    for item in text.split(","):
        name, rest = item.split("=")
        host, port, mbps, rtt = rest.split(":")
        out.append({"name": name, "addr": f"{host}:{port}", "mbps": float(mbps),
                    "rtt_ms": float(rtt)})
    return out


def run_users(runner, link, n_users, q, prompts, pool, split, k_cfg, mask_id):
    """N users at once; returns one dict per answer (text still as ids)."""
    barrier = threading.Barrier(n_users)
    rows, errors, lock = [], [], threading.Lock()

    def user(u):
        sock = None
        try:
            sock = connect(link["addr"], wait_s=120)
            sock.settimeout(900)
            stream = torch.cuda.Stream(runner.device)
            with torch.cuda.stream(stream):
                barrier.wait(timeout=300)
                off = u * pool // n_users
                for j in range(q):
                    qi = (off + j) % pool
                    case, ids = prompts[qi]
                    c = session_cfg("fastdllm", split, "bf16", case["case_id"])
                    c.update(user=u, users=n_users, link=link["name"], config=k_cfg)
                    t0 = time.time()
                    answer, info = runner.session(sock, c, ids, mask_id)
                    with lock:
                        rows.append({"user": u, "j": j, "question": qi, "case_id": case["case_id"],
                                     "start": t0, "end": time.time(), "answer_ids": answer,
                                     **info})
        except BaseException as exc:   # noqa: BLE001
            with lock:
                errors.append((u, repr(exc)))
            try:
                barrier.abort()
            except Exception:  # noqa: BLE001
                pass
        finally:
            if sock is not None:
                try:
                    send_frame(sock, F_BYE)
                except OSError:
                    pass
                sock.close()

    threads = [threading.Thread(target=user, args=(u,), name=f"user{u}") for u in range(n_users)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError(f"{len(errors)} user(s) failed: {errors[:3]}")
    return rows


def user_rate(g):
    """Answers per hour: each user's answers divided by that user's active time (first
    prompt sent to last answer held; users ask again at once), summed over the users."""
    per_user = g.groupby("user").agg(n=("end", "size"), t0=("start", "min"), t1=("end", "max"))
    return float((per_user["n"] / (per_user["t1"] - per_user["t0"])).sum() * 3600)


def summarize(df):
    alone = (df[df["users"] == 1].set_index(["config", "link", "case_id"])["text"].to_dict())
    df = df.copy()
    df["same_as_one_user"] = [alone.get((c, l, q)) == t if (c, l, q) in alone else np.nan
                              for c, l, q, t in zip(df["config"], df["link"], df["case_id"],
                                                    df["text"])]
    out = []
    for key, g in df.groupby(KEYS, sort=False):
        span = g["end"].max() - g["start"].min()
        same = g["same_as_one_user"].dropna()
        out.append({**dict(zip(KEYS, key)), "split": int(g["split"].iloc[0]),
                    "answers": len(g), "acc": g["ok"].mean(),
                    "answers_per_hour": user_rate(g),
                    "answers_per_hour_whole_run": len(g) / span * 3600,
                    "latency_s": g["latency_s"].mean(), "latency_median_s": g["latency_s"].median(),
                    "latency_p90_s": g["latency_s"].quantile(0.9),
                    "steps": g["n_steps"].mean(), "up_mb": g["edge_up_bytes"].mean() / 1e6,
                    "edge_s": g["edge_compute_ms"].mean() / 1e3,
                    "edge_with_slowdown_s": (g["edge_compute_ms"] + g["edge_sleep_ms"]).mean() / 1e3,
                    "cloud_gen_s": g["cloud_gen_s"].mean(),
                    "wire_s": g["wire_ms"].mean() / 1e3,
                    "same_as_one_user": same.mean() if len(same) else np.nan})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="label, e.g. split4 or cloud_only")
    ap.add_argument("--split", type=int, required=True, help="0 = cloud-only")
    ap.add_argument("--links", required=True, help="name=host:port:mbps:rtt_ms,...")
    ap.add_argument("--users", default="1,2,4,8,12,16")
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--min-q", type=int, default=6)
    ap.add_argument("--slowdown", type=float, default=3.0)
    ap.add_argument("--tag", default="lab")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from exp9b_official_feature_cache_diagnostics import prepare_input
    from exp10d_gsm8k_refresh import gsm8k_score, load_cases
    out_dir = ROOT / "outputs" / f"exp22_multiuser_{a.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.csv"
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mask_id = int(tok.mask_token_id)
    cases = load_cases(n=a.pool, start=0)
    prompts = [(c, prepare_input(tok, c["prompt"], "cpu")[0][0].tolist()) for c in cases]
    dev = torch.device(a.device)
    models = {}
    if a.split > 0:
        repo = Path(FAST_REPO)
        models["fastdllm"] = load_edge_model("fastdllm", MODEL_NAME, a.split, dev,
                                             repo if repo.is_absolute() else (ROOT / repo))
    runner = EdgeRunner(models, dev, slowdown=a.slowdown if a.split > 0 else 1.0)
    links = parse_links(a.links)
    users = [int(u) for u in a.users.split(",")]
    old = pd.read_csv(runs_path) if runs_path.is_file() else pd.DataFrame()
    print(f"Exp22 edge | {torch.cuda.get_device_name(dev)} | {a.config} (split {a.split}) | "
          f"links {[l['name'] for l in links]} | users {users} | slowdown {a.slowdown:g}",
          flush=True)
    for link in links:
        for n in users:
            q = max(a.pool // n, a.min_q)
            if len(old) and ((old["config"] == a.config) & (old["link"] == link["name"])
                             & (old["users"] == n)).sum() >= n * q:
                print(f"  {link['name']} users={n}: already done", flush=True)
                continue
            # warm-up: one answer over this link before the timed users start
            run_users(runner, link, 1, 1, prompts, a.pool, a.split, a.config, mask_id)
            t0 = time.time()
            rows = run_users(runner, link, n, q, prompts, a.pool, a.split, a.config, mask_id)
            for r in rows:
                case = prompts[r["question"]][0]
                text = describe(tok, r.pop("answer_ids"))["text"]
                r.update({"config": a.config, "split": a.split, "link": link["name"],
                          "mbps": link["mbps"], "rtt_ms": link["rtt_ms"], "users": n,
                          "slowdown": runner.slow, "latency_s": r["edge_wall_s"],
                          "ok": bool(gsm8k_score(case, text)["ok"]), "text": text})
            new = pd.DataFrame(rows).sort_values(["start"])
            if len(old):   # drop a partial earlier attempt of the same combination
                old = old[~((old["config"] == a.config) & (old["link"] == link["name"])
                            & (old["users"] == n))]
            old = pd.concat([old, new], ignore_index=True)
            old.to_csv(runs_path, index=False)
            s = summarize(new).iloc[0]
            print(f"  {time.strftime('%F %T')} {link['name']:<9s} users={n:2d}: {len(new):3d} answers "
                  f"in {time.time() - t0:6.1f} s | {s['answers_per_hour']:7.1f} answers/h | latency "
                  f"{s['latency_s']:6.2f} s (median {s['latency_median_s']:6.2f}, p90 "
                  f"{s['latency_p90_s']:6.2f}) | steps {s['steps']:5.1f} | acc {s['acc']:.2f}",
                  flush=True)
    df = pd.read_csv(runs_path)
    summary = summarize(df)
    summary.to_csv(out_dir / "summary.csv", index=False)
    meta_path = out_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    meta.update({"experiment": "Exp22 several users on one cloud GPU (Fast-dLLM, bf16)",
                 "tag": a.tag, "pool": a.pool, "min_q": a.min_q, "model": MODEL_NAME,
                 "torch": torch.__version__, "transformers": transformers.__version__,
                 "python": platform.python_version(), "host": platform.node()})
    meta.setdefault("configs", {})[a.config] = {
        "split": a.split, "links": links, "users": users, "slowdown": runner.slow,
        "edge_gpu": torch.cuda.get_device_name(dev)}
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"\nExp22 edge {a.config} completed. Output: {out_dir}")
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
