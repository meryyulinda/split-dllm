"""Thin runners around pinned author code. No cache method is rebranded as DASR.

Run in a fresh process/env for each repository (their imports can conflict).
Full GPU weights must fit. Split composition is not implemented by this runner.
"""
import argparse
import importlib.util
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=("fast_v1", "dllm_cache"))
    p.add_argument("--repo", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--threshold", type=float)
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--prompt-interval", type=int, default=100)
    p.add_argument("--generation-interval", type=int, default=7)
    p.add_argument("--transfer-ratio", type=float, default=.25)
    args = p.parse_args()
    import torch
    from transformers import AutoModel, AutoTokenizer
    from dasr.metrics import environment, score_numeric, write_json, bootstrap
    from dasr.__main__ import examples
    if not torch.cuda.is_available():
        raise RuntimeError("Author-code timing baselines in this runner require a GPU")
    repo = Path(args.repo).resolve()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if args.method == "fast_v1":
        module_root = repo / "v1" / "llada"
        if not module_root.exists():
            raise FileNotFoundError("Use current NVlabs/Fast-dLLM layout with v1/llada")
        sys.path.insert(0, str(module_root))
        from model.modeling_llada import LLaDAModelLM
        from generate import generate_with_prefix_cache
        cls, sampler = LLaDAModelLM, generate_with_prefix_cache
    else:
        sys.path.insert(0, str(repo))
        from utils import generate
        from dllm_cache.cache import dLLMCache, dLLMCacheConfig
        from dllm_cache.hooks import register_cache_LLaDA
        cls, sampler = AutoModel, generate
    model = cls.from_pretrained(args.checkpoint, revision=args.revision, trust_remote_code=True,
                                torch_dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True, device_map={"": "cuda:0"}).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, revision=args.revision, trust_remote_code=True)
    if args.method == "dllm_cache":
        from dataclasses import asdict
        dLLMCache.new_instance(**asdict(dLLMCacheConfig(prompt_interval_steps=args.prompt_interval,
                                                      gen_interval_steps=args.generation_interval,
                                                      transfer_ratio=args.transfer_ratio)))
        register_cache_LLaDA(model, "model.transformer.blocks")
    rows = examples(args.data, args.limit)
    result_path = Path(args.output)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for i, row in enumerate([rows[0]] * args.warmup + rows):
        text = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}], add_generation_prompt=True, tokenize=False)
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        if args.method == "dllm_cache":
            dLLMCache().reset_cache(tokens.input_ids.shape[1])
        options = dict(steps=args.steps, gen_length=args.gen_length, block_length=args.block_length, temperature=0.)
        if args.method == "fast_v1":
            options["threshold"] = args.threshold
            call = lambda: sampler(model, tokens.input_ids, **options)
        else:
            if args.threshold is not None:
                raise ValueError("dLLM-Cache runner holds its native fixed sampler; threshold is a Fast-dLLM factor")
            # Refuse silently changed author interfaces.
            if "attention_mask" not in inspect.signature(sampler).parameters:
                raise RuntimeError("Pinned dLLM-Cache sampler interface changed; inspect utils.py")
            call = lambda: sampler(model=model, input_ids=tokens.input_ids, attention_mask=tokens.attention_mask, **options)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            output = call()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        if isinstance(output, tuple):
            output, nfe = output
        else:
            nfe = None  # Author code does not expose NFE; never invent it.
        decoded = tokenizer.decode(output[0, tokens.input_ids.shape[1]:], skip_special_tokens=True)
        if i >= args.warmup:
            results.append(dict(id=row["id"], text=decoded, latency_ms=elapsed, nfe=nfe,
                                numeric_score=score_numeric(decoded, row.get("answer")),
                                gpu_peak_bytes=torch.cuda.max_memory_allocated()))
            print(args.method, row["id"], elapsed, flush=True)
    with result_path.open("w", encoding="utf-8") as sink:
        for row in results:
            sink.write(json.dumps(row) + "\n")
    write_json(result_path.with_suffix(".manifest.json"), dict(arguments=vars(args), repository_commit=commit,
                                                              environment=environment(), comparison_scope="standalone_cloud_only",
                                                              sampler_changed=args.threshold is not None,
                                                              latency_ms=bootstrap([r["latency_ms"] for r in results])))


if __name__ == "__main__":
    main()
