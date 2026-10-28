import os
import sys

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.llada_generation import generate_with_history

MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
DEVICE = "cuda"

STEPS = 64
GEN_LENGTH = 64
BLOCK_LENGTH = 64


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to(DEVICE).eval()


prompt = "What is the capital of Australia?"

messages = [
    {
        "role": "user",
        "content": prompt,
    }
]

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)

encoded = tokenizer(
    formatted_prompt,
    add_special_tokens=False,
    return_tensors="pt",
)

input_ids = encoded["input_ids"].to(DEVICE)
attention_mask = encoded["attention_mask"].to(DEVICE)


output, token_history, transition_history = generate_with_history(
    model=model,
    prompt=input_ids,
    attention_mask=attention_mask,
    steps=STEPS,
    gen_length=GEN_LENGTH,
    block_length=BLOCK_LENGTH,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
)


generated_ids = output[:, input_ids.shape[1]:]


print(
    tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )
)

print()
print("Token states     :", len(token_history))
print("Transitions      :", len(transition_history))

print(
    "Initial masks    :",
    (
        token_history[0][
            :,
            input_ids.shape[1]:
        ]
        == 126336
    )
    .sum()
    .item(),
)

print(
    "Final masks      :",
    (
        token_history[-1][
            :,
            input_ids.shape[1]:
        ]
        == 126336
    )
    .sum()
    .item(),
)

print()
print("First 5 transitions:")

for row in transition_history[:5]:
    print(row)

print()
print("Last 5 transitions:")

for row in transition_history[-5:]:
    print(row)