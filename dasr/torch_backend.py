"""Explicit LLaDA/Dream adapters, streaming only owned safetensors parameters.

These adapters must pass `validate-model` on the actual checkpoint/hardware before
publication. No AR AutoModelForCausalLM substitution; native bidirectional blocks
and Dream's sampler logit shift are preserved.
"""
import gc
import json
import re
from pathlib import Path
import numpy as np
import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import AutoConfig, AutoModel, AutoTokenizer


class RowCache(torch.nn.Module):
    """Approximate token-local FFN reuse; attention remains dense.

    This actually executes the wrapped module only on selected rows. It is not a
    sparse-attention implementation and does not inherit any SOTA quality claim.
    """
    def __init__(self, module, owner):
        super().__init__()
        self.inner = module
        object.__setattr__(self, "owner", owner)
        self.cached = None

    def forward(self, x):
        owner = self.owner
        flat = x.reshape(-1, x.shape[-1])
        dense = owner.anchor or self.cached is None or len(self.cached) != len(flat) or len(owner.active) == len(flat)
        indices = torch.as_tensor(owner.active, device=x.device, dtype=torch.long)
        if not dense and owner.audit_rows:
            audit_idx = torch.as_tensor(owner.audit_rows, device=x.device, dtype=torch.long)
            audit = self.inner(flat[audit_idx])
            err = torch.linalg.vector_norm((audit - self.cached[audit_idx]).float())
            err /= torch.linalg.vector_norm(audit.float()) + 1e-12
            owner.max_audit_error = max(owner.max_audit_error, float(err.item()))
            owner.audit_executed_rows += len(audit_idx)
            if err > owner.audit_threshold:
                dense = True
                owner.audit_refreshes += 1
        if dense:
            result = self.inner(x)
            self.cached = result.reshape(-1, result.shape[-1]).detach().clone()
            selected = len(flat)
        else:
            if len(indices):
                self.cached[indices] = self.inner(flat[indices])
            result = self.cached.view(*x.shape[:-1], self.cached.shape[-1])
            selected = len(indices)
        owner.selected_rows += selected
        owner.total_rows += len(flat)
        return result


class TorchModel:
    def __init__(self, family, checkpoint, revision, split, edge_device="cpu", cloud_device="cuda:0",
                 dtype="bfloat16", role="both", selective=False, audit_threshold=.1):
        if family not in ("llada", "dream"):
            raise ValueError("Only explicit LLaDA and Dream adapters are supported")
        self.family, self.split, self.role = family, split, role
        self.edge_device, self.cloud_device = torch.device(edge_device), torch.device(cloud_device)
        self.torch_dtype = getattr(torch, dtype)
        self.logical_dtype = dtype
        if self.torch_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Use native float16, bfloat16 or float32; weight quantization is a separate experiment")
        for device in {self.edge_device, self.cloud_device}:
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable. Select CPU or a Colab GPU runtime")
            if device.type == "cuda" and dtype == "bfloat16":
                major, _ = torch.cuda.get_device_capability(device)
                if major < 8:
                    raise RuntimeError("This GPU lacks native BF16 support. Use --dtype float16 consistently for all baselines")
        self.config = AutoConfig.from_pretrained(checkpoint, revision=revision, trust_remote_code=True)
        self.revision = getattr(self.config, "_commit_hash", None) or revision
        self.config.use_cache = False
        if family == "llada":
            if getattr(self.config, "block_group_size", 1) != 1:
                raise NotImplementedError("Grouped LLaDA blocks need a separate adapter")
            self.layers, self.width = self.config.n_layers, self.config.d_model
            self.mask_id = 126336
        else:
            self.layers, self.width = self.config.num_hidden_layers, self.config.hidden_size
            self.mask_id = self.config.mask_token_id
        if not 0 <= split <= self.layers:
            raise ValueError("Split outside model layer range")
        self.vocab = self.config.vocab_size
        with init_empty_weights():
            self.model = AutoModel.from_config(self.config, trust_remote_code=True, torch_dtype=self.torch_dtype)
        self.model.eval()
        if family == "llada":
            core = self.model.model
            self.blocks = list(core.transformer.blocks)
            self.emb, self.norm = core.transformer.wte, core.transformer.ln_f
            self.head = core.transformer.wte if core.config.weight_tying else core.transformer.ff_out
            self.core_config = core.config
        else:
            self.blocks = list(self.model.model.layers)
            self.emb, self.norm, self.head = self.model.model.embed_tokens, self.model.model.norm, self.model.lm_head
            self.rotary = self.model.model.rotary_emb
        root = Path(checkpoint)
        if not root.is_dir():
            from huggingface_hub import snapshot_download
            root = Path(snapshot_download(checkpoint, revision=self.revision,
                                          allow_patterns=["*.safetensors", "*.json", "*.py"]))
        files = sorted(root.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError("Use a safetensors checkpoint directory; pickle checkpoints are not supported")
        required = {name: self.parameter_device(name) for name, _ in self.model.named_parameters()}
        required = {name: dev for name, dev in required.items() if dev is not None}
        loaded = set()
        for file in files:
            with safe_open(str(file), framework="pt", device="cpu") as shard:
                for key in shard.keys():
                    if key in required:
                        tensor = shard.get_tensor(key)
                        set_module_tensor_to_device(self.model, key, required[key], value=tensor, dtype=self.torch_dtype)
                        loaded.add(key)
                        del tensor
            gc.collect()
        missing = sorted(set(required) - loaded)
        if missing:
            raise RuntimeError("Checkpoint layout mismatch or tied-weight alias not handled: " + ", ".join(missing[:10]))
        self.audit_threshold = audit_threshold
        self.row_modules = []
        if selective:
            for i, block in enumerate(self.blocks):
                if not self.owns_layer(i):
                    continue
                names = ("ff_proj", "ff_out") if family == "llada" else ("mlp",)
                for name in names:
                    wrapped = RowCache(getattr(block, name), self)
                    setattr(block, name, wrapped)
                    self.row_modules.append(wrapped)
        self.reset()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=self.revision, trust_remote_code=True)
        self.weight_bytes_by_device = {}
        for param in self.model.parameters():
            if param.device.type != "meta":
                key = str(param.device)
                self.weight_bytes_by_device[key] = self.weight_bytes_by_device.get(key, 0) + param.numel() * param.element_size()

    def owns_layer(self, i):
        return self.role == "both" or (self.role == "edge" and i < self.split) or (self.role == "cloud" and i >= self.split)

    def parameter_device(self, name):
        pattern = r"model\.transformer\.blocks\.(\d+)\." if self.family == "llada" else r"model\.layers\.(\d+)\."
        match = re.match(pattern, name)
        if match:
            i = int(match.group(1))
            if not self.owns_layer(i):
                return None
            return self.edge_device if i < self.split else self.cloud_device
        embedding = "model.transformer.wte." if self.family == "llada" else "model.embed_tokens."
        if name.startswith(embedding):
            tied = self.family == "llada" and getattr(self.config, "weight_tying", False)
            if tied and self.role != "both":
                raise NotImplementedError("Distributed tied embeddings need explicit duplication")
            if self.role == "cloud" and self.split > 0:
                return None
            return self.cloud_device if self.split == 0 else self.edge_device
        if self.role == "edge" and self.split < self.layers:
            return None
        return self.edge_device if self.split == self.layers else self.cloud_device

    def reset(self):
        self.active, self.audit_rows, self.anchor = [], [], True
        self.selected_rows = self.total_rows = self.audit_refreshes = self.audit_executed_rows = 0
        self.max_audit_error = 0.
        for module in self.row_modules:
            module.cached = None
        for device in {self.edge_device, self.cloud_device}:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

    def configure(self, active, anchor, audit_rows=()):
        self.active, self.anchor, self.audit_rows = list(active), anchor, list(audit_rows)

    @torch.inference_mode()
    def blocks_forward(self, hidden, start, stop):
        device = hidden.device
        positions = torch.arange(hidden.shape[1], device=device)
        if self.family == "dream":
            position_ids = positions.unsqueeze(0)
            position_embeddings = self.rotary(hidden, position_ids)
        for i in range(start, stop):
            if self.family == "llada":
                hidden = self.blocks[i](hidden, attention_bias=None, layer_past=None, use_cache=False)[0]
            else:
                hidden = self.blocks[i](hidden, attention_mask="full", position_ids=position_ids,
                                        past_key_value=None, use_cache=False, output_attentions=False,
                                        cache_position=positions, position_embeddings=position_embeddings)[0]
        return hidden

    @torch.inference_mode()
    def prefix(self, ids, split, active=None, anchor=True):
        if split != self.split or (self.role == "cloud" and split > 0):
            raise ValueError("Partition was not loaded for this prefix")
        device = self.cloud_device if split == 0 else self.edge_device
        ids = torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        hidden = self.emb(ids)
        if self.family == "llada":
            if self.core_config.input_emb_norm:
                hidden = hidden * self.width ** .5
        return self.blocks_forward(hidden, 0, split)

    @torch.inference_mode()
    def suffix(self, hidden, split, active=None, anchor=True):
        if split != self.split:
            raise ValueError("Partition was not loaded for this suffix")
        device = self.edge_device if split == self.layers else self.cloud_device
        hidden = hidden.to(device)
        hidden = self.norm(self.blocks_forward(hidden, split, self.layers))
        if self.family == "llada" and self.core_config.weight_tying:
            logits = torch.nn.functional.linear(hidden, self.head.weight.to(device))
        else:
            logits = self.head(hidden)
        if self.family == "llada" and self.core_config.scale_logits:
            logits = logits / self.width ** .5
        return logits[0]

    @torch.inference_mode()
    def proposals(self, logits):
        if self.family == "dream":
            # Dream predicts shifted positions; forgetting this silently corrupts sampling.
            logits = torch.cat((logits[:1], logits[:-1]), dim=0)
        ids = logits.argmax(dim=-1)
        probs = torch.softmax(logits if self.family == "llada" else logits.float(), dim=-1)
        confidence = probs.gather(-1, ids[:, None]).squeeze(-1)
        if self.family == "dream":
            confidence = confidence.to(logits.dtype)
        return ids.cpu().numpy(), confidence.float().cpu().numpy()

    def to_wire(self, hidden):
        raw = hidden.detach().contiguous().cpu()
        if self.torch_dtype == torch.bfloat16:
            return raw.view(torch.uint16).numpy()[0].copy(), "bfloat16"
        return raw.numpy()[0].copy(), self.logical_dtype

    def from_wire(self, hidden, logical_dtype):
        raw = torch.from_numpy(hidden.copy())
        if logical_dtype == "bfloat16":
            raw = raw.view(torch.bfloat16)
        return raw.unsqueeze(0)

    def sync(self, side=None):
        devices = {self.edge_device, self.cloud_device} if side is None else {
            self.edge_device if side == "edge" and self.split > 0 else self.cloud_device}
        for device in devices:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    def memory(self):
        out = {"weight_bytes_by_device": self.weight_bytes_by_device}
        for side, device in (("edge", self.edge_device), ("cloud", self.cloud_device)):
            out[side + "_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            out[side + "_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        try:
            import psutil
            out["process_rss_bytes"] = psutil.Process().memory_info().rss
        except ImportError:
            out["process_rss_bytes"] = None
        return out

    def tokenize(self, prompt):
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                 add_generation_prompt=True, tokenize=False)
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)
