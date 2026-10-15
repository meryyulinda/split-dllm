"""Colab sizing, pinned downloads, and resource guards."""
import json
from pathlib import Path


def hardware():
    import psutil
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("In Colab choose Runtime > Change runtime type > GPU")
    free, total = torch.cuda.mem_get_info()
    major, _ = torch.cuda.get_device_capability()
    return dict(gpu=torch.cuda.get_device_name(), gpu_free_gib=free / 2**30,
                gpu_total_gib=total / 2**30, ram_available_gib=psutil.virtual_memory().available / 2**30,
                dtype="bfloat16" if major >= 8 else "float16", torch=torch.__version__)


def choose_split(checkpoint, revision, dtype, gen_length=32, prompt_tokens=128):
    import torch
    from transformers import AutoConfig, AutoModel
    from accelerate import init_empty_weights
    config = AutoConfig.from_pretrained(checkpoint, revision=revision, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModel.from_config(config, trust_remote_code=True, torch_dtype=getattr(torch, dtype))
    params = [(name, p.numel() * 2) for name, p in model.named_parameters()]
    import re
    layers = config.n_layers
    info = hardware()
    plans = []
    for split in range(0, layers + 1, 4):
        edge = cloud = 0
        for name, size in params:
            match = re.match(r"model\.transformer\.blocks\.(\d+)\.", name)
            belongs_edge = int(match.group(1)) < split if match else name.startswith("model.transformer.wte.") and split > 0
            if split == layers:
                belongs_edge = True
            if belongs_edge:
                edge += size
            else:
                cloud += size
        # Conservative reserve for logits, kernels, codec copies and one streamed tensor.
        sequence = gen_length + prompt_tokens
        logit_gib = sequence * config.vocab_size * 8 / 2**30  # softmax temporaries
        required_gpu = cloud / 2**30 + 2.0 + logit_gib
        required_ram = edge / 2**30 + 2.5
        feasible = required_gpu < info["gpu_free_gib"] and required_ram < info["ram_available_gib"]
        plans.append(dict(split=split, edge_weight_gib=edge / 2**30, cloud_weight_gib=cloud / 2**30,
                          required_gpu_gib=required_gpu, required_ram_gib=required_ram, feasible=feasible))
    candidates = [p for p in plans if p["feasible"] and 0 < p["split"] < layers]
    if not candidates:
        raise RuntimeError("No native-precision split passes conservative sizing. Use a larger GPU/high-RAM runtime. " + json.dumps(plans))
    # Minimize CPU prefix work among memory-feasible candidates; test other splits later.
    selected = min(candidates, key=lambda p: p["split"])
    del model
    return info, selected, plans


def download_pinned(model_id, folder=None):
    from huggingface_hub import HfApi, snapshot_download
    from dasr.metrics import write_json
    revision = HfApi().model_info(model_id).sha
    kwargs = dict(repo_id=model_id, revision=revision, allow_patterns=["*.json", "*.py", "*.safetensors", "*.model", "*.txt"])
    if folder is not None:
        kwargs["local_dir"] = folder
    root = snapshot_download(**kwargs)
    write_json(Path(root) / "dasr_download_manifest.json", dict(model_id=model_id, revision=revision))
    return root, revision
