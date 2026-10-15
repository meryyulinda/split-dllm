import json
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np


def bootstrap(values, seed=0, samples=2000):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return dict(mean=None, ci95=[None, None], n=0)
    rng = np.random.RandomState(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(samples)])
    return dict(mean=float(x.mean()), ci95=np.percentile(means, [2.5, 97.5]).tolist(), n=len(x))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def environment():
    out = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__)
    try:
        import importlib.metadata as m
        for name in ("torch", "transformers", "accelerate", "safetensors", "datasets"):
            try:
                out[name] = m.version(name)
            except m.PackageNotFoundError:
                pass
        out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out["gpu"] = None
    return out


def score_numeric(text, answer):
    """Simple GSM8K-style numeric parser. NOT MATH equivalence or code scoring."""
    import re
    from decimal import Decimal, InvalidOperation
    if answer is None:
        return None
    def extract(s):
        s = str(s).split("####")[-1].replace(",", "")
        matches = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
        try:
            return Decimal(matches[-1]) if matches else None
        except InvalidOperation:
            return None
    predicted, gold = extract(text), extract(answer)
    return float(predicted is not None and gold is not None and predicted == gold)
