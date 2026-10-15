"""Run from the parent directory: python -m dasr --help."""
import argparse
import csv
import hashlib
import json
import socket
import time
from dataclasses import asdict, fields
from pathlib import Path
import numpy as np
from .codec import MODES
from .metrics import bootstrap, environment, score_numeric, write_json
from .runtime import RunConfig, CloudEndpoint, run_query
from .scheduler import Action, Scheduler
from .toy import ToyModel
from .transport import TCPClient, decode_message, encode_message, receive


DEFAULTS = {"llada": "GSAI-ML/LLaDA-8B-Instruct", "dream": "Dream-org/Dream-v0-Instruct-7B"}


def create_model(args, role="both", selective=None):
    if args.backend == "toy":
        return ToyModel()
    from .torch_backend import TorchModel
    return TorchModel(args.backend, args.checkpoint or DEFAULTS[args.backend], args.revision,
                      args.split, args.edge_device, args.cloud_device, args.dtype,
                      role=role, selective=args.selective if selective is None else selective,
                      audit_threshold=args.audit_threshold)


def examples(path, limit=None):
    if path:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = [dict(id="debug-%02d" % i, prompt="A shop sells 3 boxes with 4 pencils each. How many pencils?",
                     answer="12", task="smoke") for i in range(3)]
    if limit:
        rows = rows[:limit]
    if not rows or len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Supply nonempty JSONL with unique id, prompt, answer fields")
    return rows


def tokenize(model, text):
    if model.family == "toy":
        return [1 + ord(c) % (model.vocab - 1) for c in text[:12]]
    return model.tokenize(text)


def config_from(args):
    return RunConfig(**{field.name: getattr(args, field.name) for field in fields(RunConfig)})


def benchmark(args, model=None):
    configs = [dict(name=args.codec, codec=args.codec, threshold=args.threshold, refresh=args.refresh,
                    selective=args.selective, active_fraction=args.active_fraction)]
    if args.sweep:
        configs = json.loads(Path(args.sweep).read_text(encoding="utf-8"))["configurations"]
    elif args.adaptive_profile:
        configs = [dict(name="full_dense", codec="full", selective=args.selective, adaptive_profile=None),
                   dict(name="adaptive_codec", codec="xor", selective=args.selective, adaptive_profile=args.adaptive_profile)]
    if configs[0].get("codec") not in ("full", "zlib", "xor") or configs[0].get("selective", False):
        raise ValueError("First sweep configuration must be an exact dense reference")
    selective_any = any(c.get("selective", False) for c in configs)
    remote = args.transport == "tcp"
    if remote and args.split == 0:
        # Tokenizer/client model needs no full cloud checkpoint; use cloud-only local
        # interface for tokenization instead of constructing a GPU partition.
        if args.backend != "toy":
            from transformers import AutoTokenizer
            class TokenizerClient(ToyModel):
                def __init__(self):
                    super().__init__()
                    self.family = args.backend
                    self.logical_dtype = args.dtype
                    self.tokenizer = AutoTokenizer.from_pretrained(args.checkpoint or DEFAULTS[args.backend], revision=args.revision, trust_remote_code=True)
                def tokenize(self, text):
                    return self.tokenizer(self.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                                           add_generation_prompt=True, tokenize=False), add_special_tokens=False)["input_ids"]
                def decode(self, ids):
                    return self.tokenizer.decode(ids, skip_special_tokens=True)
            model = TokenizerClient()
        else:
            model = ToyModel()
    else:
        model = model or create_model(args, role="edge" if remote else "both", selective=selective_any)
    tcp = TCPClient(args.host, args.port) if remote else None
    if tcp:
        hello, _ = decode_message(tcp.exchange(encode_message(dict(op="hello"))))
        if hello["family"] != args.backend or hello["split"] != args.split or hello["logical_dtype"] != model.logical_dtype:
            raise ValueError("Client/server backend, split or native dtype mismatch")
        if args.backend != "toy" and args.split > 0 and hello["revision"] != model.revision:
            raise ValueError("Client/server checkpoint revisions differ")
    data = examples(args.data, args.limit)
    input_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", dict(environment=environment(), arguments=vars(args), input_sha256=input_hash,
                                         checkpoint_revision=getattr(model, "revision", args.revision),
                                         deployment="physical_tcp" if remote else "single_host_emulation",
                                         native_model_validated=bool(args.validation_report),
                                         validation_report=args.validation_report,
                                         wire_byte_scope="application_frames_including_telemetry"))
    if args.validation_report:
        validation = json.loads(Path(args.validation_report).read_text(encoding="utf-8"))
        if (not validation.get("official_sampler_output_identical") or
                not all(check["bitwise_logits_equal"] for check in validation.get("checks", [])) or
                validation.get("checkpoint_revision") != getattr(model, "revision", args.revision) or
                validation.get("dtype") != args.dtype):
            raise ValueError("Validation report does not certify this checkpoint's native forward/sampler parity")
    base_outputs = {}
    summaries = []
    try:
        for setting in configs:
            cfg = config_from(args)
            for key, value in setting.items():
                if key != "name":
                    if not hasattr(cfg, key):
                        raise ValueError("Unknown sweep field: " + key)
                    setattr(cfg, key, value)
            cfg.validate(model.family)
            name = setting["name"]
            if not name or any(c in name for c in ("/", "\\", ":")):
                raise ValueError("Invalid configuration filename")
            # Installed wrappers also honor a dense configuration by using all rows.
            for warmup in range(args.warmup):
                run_query(model, tokenize(model, data[warmup % len(data)]["prompt"]), cfg, tcp)
            results = []
            with (out / (name + ".jsonl")).open("w", encoding="utf-8") as sink:
                for row in data:
                    result = run_query(model, tokenize(model, row["prompt"]), cfg, tcp)
                    result.update(id=row["id"], task=row.get("task", "custom"), config=asdict(cfg),
                                  systems_only=model.family == "toy", reference_agreement=None, numeric_score=None)
                    if hasattr(model, "decode"):
                        result["text"] = model.decode(result["output_ids"])
                        result["numeric_score"] = score_numeric(result["text"], row.get("answer")) if row.get("scorer", "numeric") == "numeric" else None
                    if name == configs[0]["name"]:
                        base_outputs[str(row["id"])] = result["output_ids"]
                    reference = base_outputs[str(row["id"])]
                    result["reference_agreement"] = float(np.mean(np.asarray(reference) == result["output_ids"]))
                    exact_stream = not cfg.selective and all(s.get("codec_mode", "tokens_only") in
                                    ("full", "zlib", "xor", "tokens_only") for s in result["steps"])
                    if exact_stream:
                        boundaries = [s["boundary_bitwise_equal"] for s in result["steps"] if s["boundary_bitwise_equal"] is not None]
                        if not all(boundaries) or result["output_ids"] != reference:
                            raise AssertionError("Exact codec failed output or boundary identity. Stop this sweep")
                    sink.write(json.dumps(result, allow_nan=False) + "\n")
                    sink.flush()
                    results.append(result)
                    print(name, row["id"], "%.1f ms, %d application bytes" %
                          (result["totals"]["latency_ms"], result["totals"]["application_bytes"]), flush=True)
            summary = dict(name=name, config=asdict(cfg), measured_queries=len(results),
                           publication_protocol_ready=len(results) >= 30 and model.family != "toy" and bool(args.validation_report) and not cfg.trace_drift,
                           latency_ms=bootstrap([r["totals"]["latency_ms"] for r in results]),
                           modeled_latency_ms=bootstrap([r["totals"]["modeled_latency_ms"] for r in results]),
                           application_bytes=bootstrap([r["totals"]["application_bytes"] for r in results]),
                           token_agreement=bootstrap([r["reference_agreement"] for r in results]),
                           numeric_score=bootstrap([r["numeric_score"] for r in results if r["numeric_score"] is not None]))
            summaries.append(summary)
        write_json(out / "summary.json", summaries)
    finally:
        if tcp:
            tcp.close()
    return summaries


def serve(args):
    model = create_model(args, role="cloud")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(1)
        print("Cloud partition listening on %s:%d" % (args.host, args.port), flush=True)
        while True:
            conn, _ = listener.accept()
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                endpoint = CloudEndpoint(model, args.split)
                while True:
                    try:
                        message = receive(conn)
                        conn.sendall(endpoint.exchange(message))
                    except ConnectionError:
                        break


def dataset(args):
    from datasets import load_dataset
    rows = load_dataset("openai/gsm8k", "main", split="test", revision=args.revision)
    chosen = np.random.RandomState(args.seed).permutation(len(rows))[:args.limit]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as sink:
        for i in chosen:
            item = rows[int(i)]
            sink.write(json.dumps(dict(id="gsm8k-%d" % i, prompt=item["question"] +
                                       "\nSolve step by step. End with #### followed by the numerical answer.",
                                       answer=item["answer"], task="gsm8k", scorer="numeric")) + "\n")
    write_json(path.with_suffix(".manifest.json"), dict(dataset="openai/gsm8k", revision=args.revision,
                                                        fingerprint=rows._fingerprint, split="test", indices=chosen.tolist(),
                                                        seed=args.seed, fewshot=0, protocol="zero-shot numeric pilot"))


def analyze(args):
    summaries = json.loads((Path(args.input) / "summary.json").read_text(encoding="utf-8"))
    out = Path(args.input)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as sink:
        writer = csv.writer(sink)
        writer.writerow(["configuration", "queries", "latency_ms", "modeled_latency_ms", "application_bytes", "numeric_score", "agreement"])
        for s in summaries:
            writer.writerow([s["name"], s["measured_queries"]] + [s[m]["mean"] for m in
                             ("latency_ms", "modeled_latency_ms", "application_bytes", "numeric_score", "token_agreement")])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for s in summaries:
        y = s["token_agreement"]["mean"]
        for ax, metric in zip(axes, ("modeled_latency_ms", "application_bytes")):
            x = s[metric]["mean"]
            low, high = s[metric]["ci95"]
            ax.errorbar(x, y, xerr=[[x - low], [high - x]], fmt="o")
            ax.annotate(s["name"], (x, y), fontsize=8)
            ax.set_xlabel(metric + " (mean and 95% bootstrap CI)")
            ax.set_ylabel("Token agreement with dense split reference")
    figure.suptitle("Deployment pilot — agreement is not benchmark quality")
    figure.tight_layout()
    figure.savefig(out / "tradeoffs.png", dpi=180)
    plt.close(figure)
    if any(s["numeric_score"]["mean"] is not None for s in summaries):
        figure, ax = plt.subplots(figsize=(6, 4))
        for s in summaries:
            if s["numeric_score"]["mean"] is None:
                continue
            x, y = s["modeled_latency_ms"]["mean"], s["numeric_score"]["mean"]
            low, high = s["numeric_score"]["ci95"]
            ax.errorbar(x, y, yerr=[[y - low], [high - y]], fmt="o")
            ax.annotate(s["name"], (x, y), fontsize=8)
        ax.set_xlabel("Modeled latency, ms")
        ax.set_ylabel("Numeric pilot score, mean and 95% CI")
        figure.tight_layout()
        figure.savefig(out / "quality_latency.png", dpi=180)
        plt.close(figure)
    # Per-step/row RQ1 traces for each instrumented split. Run splits independently.
    for file in out.glob("*.jsonl"):
        first = json.loads(file.read_text(encoding="utf-8").splitlines()[0])
        trace = first["steps"]
        figure, ax = plt.subplots(figsize=(8, 3))
        for key in ("changed_ratio", "ffn_active_ratio", "boundary_active_ratio"):
            values = [s.get(key) for s in trace]
            if any(value is not None for value in values):
                ax.plot([s["step"] for s in trace], [np.nan if v is None else v for v in values], label=key)
        ax.set_xlabel("denoising step")
        ax.set_ylabel("row fraction")
        ax.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(out / (file.stem + "_active.png"), dpi=180)
        plt.close(figure)
        drift = [r.get("boundary_row_drift") for r in trace]
        if any(x is not None for x in drift):
            width = len(next(x for x in drift if x is not None))
            matrix = np.array([x if x is not None else [np.nan] * width for x in drift])
            figure, ax = plt.subplots(figsize=(8, 4))
            visual = ax.imshow(matrix, aspect="auto", interpolation="nearest")
            figure.colorbar(visual, ax=ax, label="normalized row drift")
            ax.set_xlabel("token position")
            ax.set_ylabel("denoising step")
            figure.tight_layout()
            figure.savefig(out / (file.stem + "_drift.png"), dpi=180)
            plt.close(figure)
    print("Wrote summary.csv and plots to", out)


def replay(args):
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    actions = [Action(**a) for a in profile["actions"]]
    sched = Scheduler(actions, args.memory_budget, args.risk_budget)
    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    result = []
    old = None
    for t, state in enumerate(trace):
        anchor = bool(state["anchor"])
        action = sched.choose(state["bandwidth_mbps"], state["rtt_ms"], anchor)
        switch = action.switch_ms if old is not None and old != action else 0
        feasible_oracle = min(sched.actions, key=lambda a: a.cost(state["bandwidth_mbps"], state["rtt_ms"]))
        result.append(dict(step=t, action=action.name, split=action.split, anchor=anchor,
                           predicted_ms=action.cost(state["bandwidth_mbps"], state["rtt_ms"]) + switch,
                           oracle_no_switch_lower_bound_ms=feasible_oracle.cost(state["bandwidth_mbps"], state["rtt_ms"])))
        old = action
    write_json(args.output, dict(kind="offline_trace_replay_not_live_split_migration", profile=profile, trace=result))


def validate_model(args):
    if args.edge_device != args.cloud_device:
        raise ValueError("Native forward validation needs edge/cloud on the same device. CPU works if GPU memory is insufficient")
    args.split = 0
    model = create_model(args, selective=False)
    if model.family == "toy":
        raise ValueError("Use unit tests for the toy backend")
    import torch
    ids = tokenize(model, "What is 2 + 3?") + [model.mask_id] * 4
    tensor = torch.tensor([ids], device=args.cloud_device)
    with torch.inference_mode():
        native_logits = model.model(tensor, use_cache=False).logits[0]
        checks = []
        for split in [0, model.layers // 4, model.layers // 2, model.layers]:
            model.split = split
            actual = model.suffix(model.prefix(ids, split), split)
            checks.append(dict(split=split, bitwise_logits_equal=torch.equal(actual, native_logits),
                               max_logit_error=float((actual - native_logits).abs().max().item())))
        model.split = 0
        cfg = config_from(args)
        cfg.split = 0
        actual = run_query(model, tokenize(model, "What is 2 + 3?"), cfg)["output_ids"]
        prompt = torch.tensor([tokenize(model, "What is 2 + 3?")], device=args.cloud_device)
        if model.family == "dream":
            official = model.model.diffusion_generate(prompt, steps=cfg.steps, max_new_tokens=cfg.gen_length,
                                                      temperature=0., alg="maskgit_plus", alg_temp=0.,
                                                      eps=1e-3, return_dict_in_generate=False)
        else:
            if not args.official_repo:
                raise ValueError("Provide --official-repo /path/to/LLaDA containing the official generate.py")
            import importlib.util
            spec = importlib.util.spec_from_file_location("official_llada_generate", Path(args.official_repo) / "generate.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            official = module.generate(model.model, prompt, steps=cfg.steps, gen_length=cfg.gen_length,
                                       block_length=cfg.block_length, temperature=0., cfg_scale=0., remasking="low_confidence")
        official_ids = official[0, len(prompt[0]):].cpu().tolist()
    report = dict(checkpoint_revision=model.revision, checks=checks,
                  official_sampler_output_identical=actual == official_ids, dtype=args.dtype,
                  environment=environment())
    write_json(args.output, report)
    if not all(c["bitwise_logits_equal"] for c in checks) or actual != official_ids:
        raise AssertionError("Native model/sampler parity failed. Inspect validation report before running experiments")
    print("Native forward and official sampler parity passed")


def profile_table(args):
    """Aggregate calibration observations; never invent latency or quality risk."""
    actions = []
    for folder in args.input:
        root = Path(folder)
        summaries = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        reference = {str(r["id"]): r for r in [json.loads(line) for line in
                     (root / (summaries[0]["name"] + ".jsonl")).read_text(encoding="utf-8").splitlines()]}
        for summary in summaries:
            rows = [json.loads(line) for line in (root / (summary["name"] + ".jsonl")).read_text(encoding="utf-8").splitlines()]
            cfg = summary["config"]
            exact = cfg["codec"] in ("full", "zlib", "xor") and not cfg["selective"]
            losses = [reference[str(r["id"])]["numeric_score"] - r["numeric_score"] for r in rows
                      if r["numeric_score"] is not None and reference[str(r["id"])]["numeric_score"] is not None]
            if not exact and len(losses) != len(rows):
                print("Skipping", summary["name"], "because held-out task scores are absent")
                continue
            risk = 0. if exact else max(0., bootstrap(losses)["ci95"][1])
            steps = [s for row in rows for s in row["steps"]]
            memory_values = []
            for row in rows:
                mem = row["totals"]["memory"]
                value = mem.get("edge_peak_allocated_bytes") or mem.get("process_rss_bytes")
                if value is not None:
                    memory_values.append(value)
            if not memory_values:
                print("Skipping", summary["name"], "because edge memory is unmeasured")
                continue
            mean = lambda key: float(np.mean([s.get(key, 0.) for s in steps]))
            action = Action("k%d_%s" % (cfg["split"], summary["name"]), cfg["split"], cfg["codec"],
                            cfg["threshold"], cfg["refresh"], mean("edge_ms"), mean("cloud_ms"),
                            mean("encode_ms"), mean("decode_ms"), mean("uplink_application_bytes"),
                            mean("downlink_application_bytes"), max(memory_values), risk, args.switch_ms)
            action.selective, action.active_fraction = cfg["selective"], cfg["active_fraction"]
            actions.append(asdict(action))
    if not actions:
        raise ValueError("No actions have measured memory and calibration quality")
    if len({a["name"] for a in actions}) != len(actions):
        raise ValueError("Duplicate split/configuration names in profile inputs")
    write_json(args.output, dict(kind="empirical_calibration_table", actions=actions,
                                calibration_inputs=args.input, switch_ms_source="user_measured",
                                notes="CPU RSS is conservative whole-process memory. Use separate held-out test data."))


def parser():
    p = argparse.ArgumentParser(description="DASR experiments: Colab emulation or physical edge/cloud RPC")
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("bench", "serve", "validate-model"):
        c = sub.add_parser(command)
        c.add_argument("--backend", choices=("toy", "llada", "dream"), default="toy")
        c.add_argument("--checkpoint")
        c.add_argument("--revision", default="main")
        c.add_argument("--split", type=int, default=2)
        c.add_argument("--edge-device", default="cpu")
        c.add_argument("--cloud-device", default="cuda:0")
        c.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
        c.add_argument("--selective", action="store_true")
        c.add_argument("--audit-threshold", type=float, default=.1)
        c.add_argument("--host", default="127.0.0.1")
        c.add_argument("--port", type=int, default=8765)
        c.add_argument("--codec", choices=MODES, default="xor")
        c.add_argument("--threshold", type=float, default=0.)
        c.add_argument("--refresh", type=int, default=4)
        c.add_argument("--active-fraction", type=float, default=.25)
        c.add_argument("--audit-fraction", type=float, default=.02)
        c.add_argument("--gen-length", type=int, default=32)
        c.add_argument("--steps", type=int, default=32)
        c.add_argument("--block-length", type=int, default=32)
        c.add_argument("--bandwidth-mbps", type=float, default=100)
        c.add_argument("--rtt-ms", type=float, default=20)
        c.add_argument("--sleep-network", action="store_true")
        c.add_argument("--seed", type=int, default=7)
        c.add_argument("--trace-drift", action="store_true")
        c.add_argument("--output", default="results/pilot")
        c.add_argument("--data")
        c.add_argument("--limit", type=int)
        c.add_argument("--warmup", type=int, default=1)
        c.add_argument("--transport", choices=("local", "tcp"), default="local")
        c.add_argument("--sweep")
        c.add_argument("--official-repo")
        c.add_argument("--validation-report")
        c.add_argument("--adaptive-profile")
        c.add_argument("--network-trace")
        c.add_argument("--memory-budget-bytes", type=float, default=12884901888)
        c.add_argument("--risk-budget", type=float, default=0.)
    c = sub.add_parser("prepare-gsm8k")
    c.add_argument("--output", default="data/gsm8k_pilot.jsonl")
    c.add_argument("--limit", type=int, default=30)
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--revision", default="main")
    c = sub.add_parser("analyze")
    c.add_argument("--input", default="results/pilot")
    c = sub.add_parser("replay-scheduler")
    c.add_argument("--profile", required=True)
    c.add_argument("--trace", required=True)
    c.add_argument("--memory-budget", type=float, required=True)
    c.add_argument("--risk-budget", type=float, default=0.)
    c.add_argument("--output", default="results/scheduler.json")
    c = sub.add_parser("profile-table")
    c.add_argument("--input", action="append", required=True)
    c.add_argument("--switch-ms", type=float, required=True, help="Measured weight-load plus cache-reset cost; not a placeholder")
    c.add_argument("--output", default="results/profile.json")
    return p


def main():
    args = parser().parse_args()
    if args.command == "bench":
        benchmark(args)
    elif args.command == "serve":
        serve(args)
    elif args.command == "prepare-gsm8k":
        dataset(args)
    elif args.command == "analyze":
        analyze(args)
    elif args.command == "replay-scheduler":
        replay(args)
    elif args.command == "validate-model":
        validate_model(args)
    elif args.command == "profile-table":
        profile_table(args)


if __name__ == "__main__":
    main()
