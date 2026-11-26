"""Measure how fast a GPU runs DASR's edge part (embedding + Dream's first k layers).

Builds Dream-v0-Instruct-7B's architecture with only k layers and RANDOM weights (timing
does not depend on the weight values), so it needs only the model's config and code from
the Hugging Face hub, about 3 GB of GPU memory and no 15-GB download. Times one denoising
step of the edge for m rows (CUDA events, median of --reps).

Run the same command on the lab's A6000 and on the rented GPU; the ratio of the two is the
measured edge slowdown s (the paper assumes 3).
  pip install "transformers==4.46.2"          # same version as the lab server
  python edge_gpu_bench.py                     # prints a table and one JSON line

Rows: 32 = Fast-dLLM in-block step (attention among the 32 rows only; the 360 cached
keys would add < 1% of the work), 256 = dLLM-Cache answer rows, 360 = full boundary on
GSM8K, 865 = full boundary on the long prompts.
"""
import argparse
import json
import platform

import torch
from transformers import AutoConfig, AutoModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--rows", type=int, nargs="+", default=[32, 256, 360, 865])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    a = ap.parse_args()
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16
    dev = torch.device("cuda")
    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True)
    cfg.num_hidden_layers = a.layers
    torch.manual_seed(0)
    with dev:
        model = AutoModel.from_config(cfg, trust_remote_code=True, torch_dtype=dtype)
    model.eval()
    base = model.model          # embed_tokens, layers[0:k], norm (as in src/split_runtime.py)

    def step(ids):
        try:
            return base(input_ids=ids, attention_mask="full")
        except (TypeError, ValueError, AttributeError):
            return base(input_ids=ids)

    res = {}
    name = torch.cuda.get_device_name(dev)
    print(f"{name} | {a.dtype} | {a.layers} layers of {a.model} (random weights) | "
          f"torch {torch.__version__}")
    with torch.inference_mode():
        for m in a.rows:
            ids = torch.randint(0, cfg.vocab_size, (1, m), device=dev)
            for _ in range(3):
                step(ids)                      # warm-up
            times = []
            for _ in range(a.reps):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                step(ids)
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            times.sort()
            res[m] = times[len(times) // 2]
            print(f"  {m:4d} rows: {res[m]:8.2f} ms per step")
    print("JSON " + json.dumps({"gpu": name, "dtype": a.dtype, "layers": a.layers,
                                "ms_per_step": res, "torch": torch.__version__,
                                "python": platform.python_version()}))


if __name__ == "__main__":
    main()
