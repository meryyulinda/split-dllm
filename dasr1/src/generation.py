import torch

def prepare_prompt(tokenizer, prompt: str):
    messages = [
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True,)
    
    return {key: value.to("cuda") for key, value in inputs.items()}

@torch.no_grad()
def generate_baseline(model, tokenizer, prompt, max_new_tokens, steps, temperature, top_p=None, top_k=None, alg="origin",):
    inputs = prepare_prompt(tokenizer, prompt)

    outputs = model.diffusion_generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        steps=steps,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        alg=alg,
        return_dict_in_generate=True,
    )

    sequence = outputs.sequences[0]

    decoded = tokenizer.decode(sequence, skip_special_tokens=True,)

    return {
        "sequence": sequence,
        "text": decoded,
        "inputs": inputs,
    }

@torch.no_grad()
def generate_with_token_history(model, tokenizer, prompt, max_new_tokens, steps, temperature, history_hook, top_p=None, top_k=None, alg="origin",):
    inputs = prepare_prompt(tokenizer, prompt)

    outputs = model.diffusion_generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        steps=steps,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        alg=alg,
        return_dict_in_generate=True,
        generation_tokens_hook_func=history_hook,
    )

    sequence = outputs.sequences[0]

    decoded = tokenizer.decode(sequence, skip_special_tokens=True,)

    return {
        "sequence": sequence,
        "text": decoded,
        "inputs": inputs,
    }
# def generate_instrumented():
#     return None

# def generate_split():
#     return None

# def generate_dasr():
#     return None