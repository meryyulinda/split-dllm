"""Exp22 cloud program: one cloud model serving several edge devices at the same time.

Like exp21_cloud.py (same protocol, src/remote_split.py), but every connection is served
in its own thread and CUDA stream, all on ONE copy of the model, so that answers for
different users run concurrently on the cloud GPU. No batching: each user's steps are
separate model calls that share the GPU. Fast-dLLM backend only (dLLM-Cache keeps its
cache in one global object per process, so two users cannot share a process).

Run from dasr1/ on the lab server (started by run_exp22.sh):
  CUDA_VISIBLE_DEVICES=1 python experiments/exp22_cloud.py --split 4 --port 6200
--split 0 = cloud-only (the cloud runs every layer; the edge only sends the prompt).

Every answer is appended to outputs/exp22_cloud_logs/fastdllm_k<split>.jsonl with its
thread, start and end times and the cloud's per-answer statistics.
"""
import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from exp10_common import ROOT  # noqa: E402
from exp21_cloud import load, run_session  # noqa: E402
from src.remote_split import (F_BEGIN, F_BYE, F_DONE, RemoteLink, ThreadLinks,  # noqa: E402
                              _tune, recv_frame, send_frame, strip_edge)


def serve(conn, peer, model, tok, router, device, split, logfile, lock):
    stream = torch.cuda.Stream(device)
    link = RemoteLink(model, device, model.config.hidden_size, hook=False) if split > 0 else None
    if router is not None:
        router.set(link)
    try:
        with torch.cuda.device(device), torch.cuda.stream(stream):
            while True:
                ftype, body, _ = recv_frame(conn)
                if ftype == F_BYE:
                    break
                if ftype != F_BEGIN:
                    raise RuntimeError(f"cloud: unexpected frame type {ftype}")
                c = json.loads(bytes(body).decode())
                t0 = time.time()
                info, _steps = run_session(c, model, tok, None, link, device, "fastdllm", conn)
                send_frame(conn, F_DONE, json.dumps(info).encode())
                rec = {"start": t0, "end": time.time(), "thread": threading.current_thread().name,
                       "peer": f"{peer[0]}:{peer[1]}", "case_id": c.get("case_id"),
                       "user": c.get("user"), "users": c.get("users"), "link": c.get("link"),
                       "encoding": c.get("encoding"),
                       **{k: v for k, v in info.items() if not k.startswith("final")}}
                with lock:
                    with logfile.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
    except (EOFError, ConnectionError, OSError) as exc:
        print(f"{time.strftime('%F %T')} {peer}: connection ended: {exc!r}", flush=True)
    except Exception as exc:   # one user's error must not stop the others
        print(f"{time.strftime('%F %T')} {peer}: ERROR {exc!r}", flush=True)
        raise
    finally:
        conn.close()
        if link is not None:
            link.active = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=4)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    device = torch.device(a.device)
    torch.cuda.set_device(device)
    name = f"fastdllm_k{a.split}"
    model, tok, _ = load("fastdllm", device)
    router = None
    if a.split > 0:
        router = ThreadLinks(model)
        strip_edge(model, a.split, router)
    logdir = ROOT / "outputs" / "exp22_cloud_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logfile, lock = logdir / f"{name}.jsonl", threading.Lock()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.host, a.port))
    srv.listen(64)
    print(f"Exp22 cloud | {name} | {torch.cuda.get_device_name(device)} | "
          f"{torch.cuda.memory_allocated(device) / 2**30:.1f} GiB of weights | "
          f"listening on {a.host}:{a.port}", flush=True)
    while True:
        conn, peer = srv.accept()
        _tune(conn)
        threading.Thread(target=serve, name=f"user-{peer[1]}", daemon=True,
                         args=(conn, peer, model, tok, router, device, a.split, logfile,
                               lock)).start()


if __name__ == "__main__":
    main()
