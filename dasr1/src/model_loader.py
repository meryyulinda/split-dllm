import torch
from transformers import AutoModel, AutoTokenizer

def load_dream_model(model_name: str):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load the model
    model = AutoModel.from_pretrained(model_name, 
                                      torch_dtype=torch.bfloat16,
                                      trust_remote_code=True)
    
    model = model.to('cuda')  # Move the model to GPU if available
    model.eval()  # Set the model to evaluation mode

    return model, tokenizer