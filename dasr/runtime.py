import time
import uuid
import zlib
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from .codec import BoundaryCache, BoundaryEncoder, CacheMismatch, numeric, unpack
from .transport import encode_message, decode_message, checked_payload


@dataclass
class RunConfig:
    split: int = 2
    codec: str = "xor"
    threshold: float = 0.
    refresh: int = 4
    active_fraction: float = .25
    selective: bool = False
    audit_fraction: float = .02
    gen_length: int = 32
    steps: int = 32
    block_length: int = 32
    bandwidth_mbps: float = 100
    rtt_ms: float = 20
    sleep_network: bool = False
    seed: int = 7
    trace_drift: bool = False
    adaptive_profile: str = None
    network_trace: str = None
    memory_budget_bytes: float = 12884901888
    risk_budget: float = 0.

    def validate(self, family):
        if self.gen_length < 1 or self.steps < 1 or self.block_length < 1:
            raise ValueError("Positive output length, steps and block length required")
        if self.refresh < 1 or not 0 <= self.active_fraction <= 1 or not 0 <= self.audit_fraction <= 1:
            raise ValueError("Invalid cache settings")
        if self.bandwidth_mbps <= 0 or self.rtt_ms < 0:
            raise ValueError("Invalid network")
        if family != "dream":
            if self.gen_length % self.block_length:
                raise ValueError("gen_length must be divisible by block_length")
            blocks = self.gen_length // self.block_length
            if self.steps % blocks or self.steps > self.gen_length:
                raise ValueError("LLaDA steps must divide across blocks and not exceed output length")
        elif self.block_length != self.gen_length:
            raise ValueError("Native Dream adapter uses full-canvas maskgit_plus; block_length must equal gen_length")


def select_next(model, ids, proposed, confidence, count, block_end):
    eligible = np.asarray(ids) == model.mask_id
    eligible[block_end:] = False
    if count < 0 or count > int(eligible.sum()):
        raise ValueError("Invalid commit budget")
    result = np.asarray(ids, dtype=np.int64).copy()
    if count:
        if model.family == "toy":
            scores = np.where(eligible, confidence, -np.inf)
            positions = np.argsort(-scores, kind="stable")[:count]
        else:
            import torch
            device = model.edge_device if model.split == model.layers else model.cloud_device
            scores = torch.as_tensor(confidence, device=device, dtype=model.torch_dtype)
            scores[~torch.as_tensor(eligible, device=device)] = -torch.inf
            positions = torch.topk(scores, count).indices.cpu().numpy()
        result[positions] = proposed[positions]
    return result


class CloudEndpoint:
    def __init__(self, model, split, reset_model=True):
        self.model, self.split = model, split
        self.reset_model = reset_model
        self.cache = BoundaryCache()
        self.session = None
        self.expected_step = 0

    def exchange(self, message):
        meta, payload = decode_message(message)
        if meta.get("op") == "hello":
            return encode_message(dict(family=self.model.family, layers=self.model.layers, split=self.split,
                                       logical_dtype=self.model.logical_dtype,
                                       revision=getattr(self.model, "revision", "toy-seed-7")))
        if meta.get("op") == "generate":
            if self.split != 0:
                raise ValueError("Request-once generation needs a cloud-only partition")
            cfg = RunConfig(**meta["config"])
            result = cloud_direct(self.model, np.frombuffer(payload, dtype="<i4").tolist(), cfg)
            output = np.asarray(result.pop("output_ids"), dtype="<i4").tobytes()
            result["payload_crc32"] = zlib.crc32(output)
            return encode_message(result, output)
        if meta["split"] != self.split:
            raise ValueError("Cloud partition does not match split")
        if self.session != meta["session"]:
            if meta["step"] != 0:
                raise ValueError("New sessions must begin at step zero")
            self.session = meta["session"]
            self.expected_step = 0
            self.cache.reset()
            if self.reset_model:
                self.model.reset()
        if meta["step"] != self.expected_step:
            raise ValueError("Out-of-order/duplicate denoising step; restart query after a lost response")
        n = meta["id_count"]
        if n < 1 or len(payload) < n * 4:
            raise ValueError("Invalid token payload")
        ids = np.frombuffer(payload[:n * 4], dtype="<i4").astype(np.int64)
        boundary = payload[n * 4:]
        decode_start = time.perf_counter()
        if self.split == 0:
            hidden = None
            version, cache_crc = -1, 0
            anchor = meta.get("compute_anchor", meta["anchor"])
        else:
            try:
                header, _ = unpack(boundary)
                if header["session"] != self.session or header["split"] != self.split:
                    raise CacheMismatch("Boundary packet session/split differs from RPC")
                raw = self.cache.apply(boundary)
            except CacheMismatch as error:
                return encode_message(dict(need_anchor=True, error=str(error)))
            hidden = self.model.from_wire(raw, self.cache.logical_dtype)
            version = self.cache.version
            cache_crc = zlib.crc32(raw.tobytes())
            anchor = meta.get("compute_anchor", meta["anchor"])
        if hasattr(self.model, "configure"):
            self.model.configure(meta["active"], anchor, meta["audit_rows"])
        self.model.sync()
        decode_ms = (time.perf_counter() - decode_start) * 1000
        cloud_start = time.perf_counter()
        if hidden is None:
            hidden = self.model.prefix(ids, 0, meta["active"], anchor)
        logits = self.model.suffix(hidden, self.split, meta["active"] if meta["selective"] else None, anchor)
        proposed, confidence = self.model.proposals(logits)
        next_ids = select_next(self.model, ids, proposed, confidence, meta["commit_count"], meta["block_end"])
        self.model.sync()
        cloud_ms = (time.perf_counter() - cloud_start) * 1000
        response_payload = next_ids.astype("<i4").tobytes()
        self.expected_step += 1
        response = dict(version=version, cache_crc32=cache_crc, decode_ms=decode_ms, cloud_ms=cloud_ms,
                        payload_crc32=zlib.crc32(response_payload), audit_error=getattr(self.model, "max_audit_error", 0.),
                        audit_refreshes=self.model.audit_refreshes, memory=self.model.memory(),
                        ffn_selected_rows=self.model.selected_rows, ffn_total_rows=self.model.total_rows)
        return encode_message(response, response_payload)


def active_rows(n, changed, fraction, audit_fraction, rng):
    # Mandatory changed rows are never dropped to satisfy a percentage cap.
    chosen = set(int(i) for i in changed)
    target = int(np.ceil(n * fraction))
    if len(chosen) < target:
        candidates = np.array([i for i in range(n) if i not in chosen], dtype=np.int64)
        chosen.update(rng.choice(candidates, min(target - len(chosen), len(candidates)), replace=False).tolist())
    count = min(n, int(np.ceil(n * audit_fraction)))
    audits = sorted(rng.choice(n, count, replace=False).tolist()) if count else []
    return sorted(chosen), audits


def commit_schedule(family, prompt_length, gen_length, steps, block_length):
    if family == "dream":
        # Remaining-count rule uses actual runtime masks, since mask IDs can be predicted.
        return [(prompt_length + gen_length, None, i) for i in range(steps)]
    blocks = gen_length // block_length
    per_block = steps // blocks
    schedule = []
    for block in range(blocks):
        base, extra = divmod(block_length, per_block)
        end = prompt_length + (block + 1) * block_length
        schedule.extend((end, base + int(i < extra), i) for i in range(per_block))
    return schedule


def run_query(model, prompt_ids, config, endpoint=None, session=None):
    config.validate(model.family)
    if not 0 <= config.split <= model.layers:
        raise ValueError("Invalid split")
    if config.adaptive_profile and config.split in (0, model.layers):
        raise ValueError("Live codec scheduling needs a nontrivial fixed split; use replay for placement adaptation")
    if config.split == 0:
        from dataclasses import asdict
        model.reset()
        endpoint = endpoint or CloudEndpoint(model, 0)
        request = encode_message(dict(op="generate", config=asdict(config)), np.asarray(prompt_ids, dtype="<i4").tobytes())
        start = time.perf_counter()
        response = endpoint.exchange(request)
        result, output = decode_message(response)
        checked_payload(result, output)
        result.pop("payload_crc32")
        result["output_ids"] = np.frombuffer(output, dtype="<i4").tolist()
        volume = len(request) + len(response)
        network_ms = config.rtt_ms + 8 * volume / (config.bandwidth_mbps * 1000)
        if config.sleep_network:
            time.sleep(network_ms / 1000)
        wall = (time.perf_counter() - start) * 1000
        result["totals"].update(latency_ms=wall, modeled_latency_ms=wall + (0 if config.sleep_network else network_ms),
                                application_bytes=volume, tokens_per_second=config.gen_length / (wall / 1000))
        result["totals"]["request_once_cloud"] = True
        result["steps"][0]["uplink_application_bytes"] = len(request)
        result["steps"][-1]["downlink_application_bytes"] = len(response)
        return result
    model.reset()
    endpoint = endpoint or CloudEndpoint(model, config.split, reset_model=False)
    session = session or uuid.uuid4().hex
    encoder = BoundaryEncoder(config.codec, config.threshold, config.refresh)
    scheduler = None
    network_trace = None
    if config.adaptive_profile:
        from .scheduler import Action, Scheduler
        profile = json.loads(Path(config.adaptive_profile).read_text(encoding="utf-8"))
        actions = [Action(**a) for a in profile["actions"] if a["split"] == config.split
                   and a.get("selective", False) == config.selective
                   and (not config.selective or a.get("active_fraction", .25) == config.active_fraction)]
        scheduler = Scheduler(actions, config.memory_budget_bytes, config.risk_budget)
    if config.network_trace:
        network_trace = json.loads(Path(config.network_trace).read_text(encoding="utf-8"))
        if not network_trace:
            raise ValueError("Empty network trace")
    last_anchor, runtime_refresh = -config.refresh, config.refresh
    rng = np.random.RandomState(config.seed)
    ids = np.full(len(prompt_ids) + config.gen_length, model.mask_id, dtype=np.int64)
    ids[:len(prompt_ids)] = prompt_ids
    previous_ids = None
    previous_boundary = None
    records = []
    total_start = time.perf_counter()
    schedule = commit_schedule(model.family, len(prompt_ids), config.gen_length, config.steps, config.block_length)
    for step, (block_end, commit_count, within_block) in enumerate(schedule):
        start = time.perf_counter()
        changed = np.arange(len(ids)) if previous_ids is None else np.flatnonzero(ids != previous_ids)
        active, audits = active_rows(len(ids), changed, config.active_fraction, config.audit_fraction, rng)
        anchor = step - last_anchor >= runtime_refresh
        bandwidth, rtt = config.bandwidth_mbps, config.rtt_ms
        if network_trace:
            state = network_trace[min(step, len(network_trace) - 1)]
            bandwidth, rtt = state["bandwidth_mbps"], state["rtt_ms"]
        action_name, predicted_ms, switch_ms = None, None, 0.
        if scheduler:
            previous_action = scheduler.current
            selected = scheduler.choose(bandwidth, rtt, anchor)
            if selected != previous_action:
                anchor = True
                switch_ms = selected.switch_ms if previous_action is not None else 0.
            encoder.mode, encoder.threshold, encoder.refresh = selected.codec, selected.threshold, selected.refresh
            runtime_refresh = selected.refresh
            action_name, predicted_ms = selected.name, selected.cost(bandwidth, rtt) + switch_ms
        if anchor:
            last_anchor = step
        if not config.selective:
            active, audits = list(range(len(ids))), []
        if hasattr(model, "configure"):
            model.configure(active, anchor, audits)
        compute_anchor = anchor
        # Approximate FFN cache is force-refreshed by current layer audits.
        model.sync()
        edge_start = time.perf_counter()
        hidden = model.prefix(ids, config.split, active if config.selective else None, anchor) if config.split else None
        model.sync()
        edge_ms = (time.perf_counter() - edge_start) * 1000
        encode_start = time.perf_counter()
        encoded = None
        raw = None
        if hidden is not None and config.split < model.layers:
            raw, logical_dtype = model.to_wire(hidden)
            encoded = encoder.encode(raw, config.split, session, changed, logical_dtype, force_anchor=anchor)
            # Any codec-generated anchor also refreshes suffix compute caches.
            anchor = anchor or encoded.anchor
        if model.family == "dream":
            import torch
            device = model.edge_device if config.split == model.layers else model.cloud_device
            times = torch.linspace(1., 1e-3, config.steps + 1, device=device)
            remaining = int(np.sum(ids == model.mask_id))
            commit_count = int(remaining * (1 - times[step + 1] / times[step])) if step < config.steps - 1 else remaining
        meta = dict(op="step", session=session, step=step, split=config.split, id_count=len(ids),
                    active=active, audit_rows=audits, anchor=anchor, compute_anchor=compute_anchor, selective=config.selective,
                    commit_count=commit_count, block_end=block_end)
        boundary = encoded.packet if encoded else b""
        message = encode_message(meta, ids.astype("<i4").tobytes() + boundary)
        encode_ms = (time.perf_counter() - encode_start) * 1000
        exchange_start = time.perf_counter()
        if config.split == model.layers:
            logits = model.suffix(hidden, config.split, active if config.selective else None, anchor)
            proposed, confidence = model.proposals(logits)
            next_ids = select_next(model, ids, proposed, confidence, commit_count, block_end)
            model.sync()
            edge_ms += (time.perf_counter() - exchange_start) * 1000
            out = next_ids.astype("<i4").tobytes()
            response = encode_message(dict(version=-1, cache_crc32=0, decode_ms=0., cloud_ms=0.,
                                           payload_crc32=zlib.crc32(out), audit_error=getattr(model, "max_audit_error", 0.),
                                           audit_refreshes=model.audit_refreshes), out)
        else:
            response = endpoint.exchange(message)
        returned, response_payload = decode_message(response)
        uplink = len(message) if config.split < model.layers else 0
        downlink = len(response) if config.split < model.layers else 0
        retries = 0
        if returned.get("need_anchor"):
            if raw is None:
                raise RuntimeError("Unexpected cloud-only cache failure")
            retry_start = time.perf_counter()
            encoded = encoder.encode(raw, config.split, session, changed, logical_dtype, force_anchor=True)
            meta["anchor"] = True
            meta["compute_anchor"] = True
            message = encode_message(meta, ids.astype("<i4").tobytes() + encoded.packet)
            encode_ms += (time.perf_counter() - retry_start) * 1000
            response = endpoint.exchange(message)
            returned, response_payload = decode_message(response)
            uplink += len(message)
            downlink += len(response)
            retries = 1
        exchange_ms = (time.perf_counter() - exchange_start) * 1000
        checked_payload(returned, response_payload)
        if encoded and (returned["version"] != encoded.version or returned["cache_crc32"] != zlib.crc32(encoder.cache.value.tobytes())):
            raise CacheMismatch("Receiver reconstruction ACK differs from sender; abort query")
        previous_ids = ids.copy()
        ids = np.frombuffer(response_payload, dtype="<i4").astype(np.int64)
        if len(ids) != len(previous_ids) or not np.array_equal(ids[:len(prompt_ids)], prompt_ids):
            raise ValueError("Invalid cloud sampler response")
        network_ms = (rtt + 8 * (uplink + downlink) / (bandwidth * 1000)) if uplink else 0.
        if config.sleep_network:
            time.sleep(network_ms / 1000)
        wall_ms = (time.perf_counter() - start) * 1000
        drift = None
        if config.trace_drift and raw is not None:
            now = numeric(raw, logical_dtype)
            if previous_boundary is not None:
                drift = (np.linalg.norm(now - previous_boundary, axis=1) /
                         (np.linalg.norm(now, axis=1) + 1e-12)).tolist()
            previous_boundary = now.copy()
        records.append(dict(step=step, edge_ms=edge_ms, encode_ms=encode_ms, exchange_ms=exchange_ms,
                            decode_ms=returned["decode_ms"], cloud_ms=returned["cloud_ms"],
                            wall_ms=wall_ms, modeled_network_ms=network_ms,
                            uplink_application_bytes=uplink, downlink_application_bytes=downlink,
                            boundary_packet_bytes=len(encoded.packet) if encoded else 0,
                            codec_mode=encoded.mode if encoded else "tokens_only", anchor=encoded.anchor if encoded else anchor,
                            compute_anchor=compute_anchor,
                            boundary_bitwise_equal=encoded.bitwise_equal if encoded else None,
                            boundary_relative_error=encoded.relative_error if encoded else None,
                            changed_ratio=len(changed) / len(ids), ffn_active_ratio=len(active) / len(ids),
                            boundary_active_ratio=encoded.active_ratio if encoded else None,
                            boundary_row_drift=drift, audit_error=returned["audit_error"],
                            audit_refreshes=returned["audit_refreshes"], retries=retries,
                            scheduler_action=action_name, scheduler_predicted_ms=predicted_ms,
                            scheduler_switch_cost_estimate_ms=switch_ms,
                            bandwidth_mbps=bandwidth, rtt_ms=rtt))
    wall_total = (time.perf_counter() - total_start) * 1000
    bytes_total = sum(r["uplink_application_bytes"] + r["downlink_application_bytes"] for r in records)
    # Baseline k=0 sends token sequence at every step in this RPC workflow; document
    # separately that a request-once cloud deployment can have less control traffic.
    total = dict(latency_ms=wall_total, modeled_latency_ms=wall_total +
                 (0 if config.sleep_network else sum(r["modeled_network_ms"] for r in records)),
                 application_bytes=bytes_total, gen_length=config.gen_length,
                 output_mask_count=int(np.sum(ids[len(prompt_ids):] == model.mask_id)),
                 tokens_per_second=config.gen_length / (wall_total / 1000),
                 ffn_selected_rows=model.selected_rows, ffn_total_rows=model.total_rows,
                 memory=model.memory(), energy_j=None, edge_flops=None, cloud_flops=None)
    if endpoint is not None and not isinstance(endpoint, CloudEndpoint) and config.split < model.layers:
        total["ffn_selected_rows"] += returned.get("ffn_selected_rows", 0)
        total["ffn_total_rows"] += returned.get("ffn_total_rows", 0)
        total["memory"]["remote_cloud"] = returned.get("memory")
    return dict(totals=total, steps=records, output_ids=ids[len(prompt_ids):].tolist())


def cloud_direct(model, prompt_ids, config):
    """Sampler and denoising remain in cloud; only one request/response crosses link."""
    config.validate(model.family)
    model.reset()
    ids = np.full(len(prompt_ids) + config.gen_length, model.mask_id, dtype=np.int64)
    ids[:len(prompt_ids)] = prompt_ids
    records = []
    previous = None
    rng = np.random.RandomState(config.seed)
    schedule = commit_schedule(model.family, len(prompt_ids), config.gen_length, config.steps, config.block_length)
    start = time.perf_counter()
    for step, (end, count, _) in enumerate(schedule):
        changed = np.arange(len(ids)) if previous is None else np.flatnonzero(ids != previous)
        active, audits = active_rows(len(ids), changed, config.active_fraction, config.audit_fraction, rng)
        if not config.selective:
            active, audits = list(range(len(ids))), []
        anchor = step % config.refresh == 0
        if hasattr(model, "configure"):
            model.configure(active, anchor, audits)
        model.sync()
        tick = time.perf_counter()
        hidden = model.prefix(ids, 0, active if config.selective else None, anchor)
        logits = model.suffix(hidden, 0, active if config.selective else None, anchor)
        proposed, confidence = model.proposals(logits)
        if model.family == "dream":
            import torch
            times = torch.linspace(1., 1e-3, config.steps + 1, device=model.cloud_device)
            remaining = int(np.sum(ids == model.mask_id))
            count = int(remaining * (1 - times[step + 1] / times[step])) if step < config.steps - 1 else remaining
        previous = ids.copy()
        ids = select_next(model, ids, proposed, confidence, count, end)
        model.sync()
        duration = (time.perf_counter() - tick) * 1000
        records.append(dict(step=step, edge_ms=0., encode_ms=0., decode_ms=0., cloud_ms=duration,
                            wall_ms=duration, uplink_application_bytes=0, downlink_application_bytes=0,
                            boundary_packet_bytes=0, anchor=anchor, boundary_bitwise_equal=None,
                            changed_ratio=len(changed) / len(ids), ffn_active_ratio=len(active) / len(ids),
                            boundary_active_ratio=None, boundary_row_drift=None,
                            audit_error=getattr(model, "max_audit_error", 0.), audit_refreshes=model.audit_refreshes))
    elapsed = (time.perf_counter() - start) * 1000
    return dict(totals=dict(latency_ms=elapsed, modeled_latency_ms=elapsed, application_bytes=0,
                            gen_length=config.gen_length, output_mask_count=int(np.sum(ids[len(prompt_ids):] == model.mask_id)),
                            tokens_per_second=config.gen_length / (elapsed / 1000),
                            ffn_selected_rows=model.selected_rows, ffn_total_rows=model.total_rows,
                            memory=model.memory(), energy_j=None, edge_flops=None, cloud_flops=None),
                steps=records, output_ids=ids[len(prompt_ids):].tolist())
