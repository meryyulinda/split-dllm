import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
from transformers import AutoModel


MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).cuda().eval()


print(model)