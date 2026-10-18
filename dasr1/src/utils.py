import random
import numpy as np
import torch
import time

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_gpu_info() -> None:
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
        print(f"GPU Memory: {total_memory:.2f} GB")
    else:
        print("No GPUs available.")

def cuda_time(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()

    result = fn()

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    return result, elapsed_ms

def boundary_size_mb(tensor):
    return (
        tensor.numel()
        * tensor.element_size()
        / (1024 ** 2)
    )
