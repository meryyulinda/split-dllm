import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_gpu_info() -> None:
    """
    Print information about the available GPUs.
    """
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
        print(f"GPU Memory: {total_memory:.2f} GB")
    else:
        print("No GPUs available.")