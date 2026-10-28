import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).cuda().eval()

print(type(model))
print("Loaded:", MODEL_NAME)