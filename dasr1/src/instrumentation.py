import torch

def count_token_changes(previous_tokens: torch.Tensor, current_tokens: torch.Tensor,):

    # Count the number of positions where tokens differ
    changed_mask = (previous_tokens != current_tokens)

    changed_count = changed_mask.sum().item()
    
    total_tokens = (changed_mask.numel())

    changed_ratio = (changed_count / total_tokens) if total_tokens > 0 else 0.0
    
    return (changed_mask, changed_count, changed_ratio)

def count_masks(tokens: torch.Tensor, mask_token_id: int, ):
    return (tokens == mask_token_id).sum().item()

def changed_indices(changed_mask: torch.Tensor):
    return (torch.where(changed_mask)[0].cpu().tolist())

def create_token_history_hook():
    history = []

    def hook(step, x, logits):
        history.append(x[0].detach().clone().cpu())

        return x
 
    return hook, history


