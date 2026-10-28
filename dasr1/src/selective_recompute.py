import torch


@torch.no_grad()
def build_selective_boundary(
    current_boundary,
    previous_boundary,
    threshold,
    eps=1e-8,
):
    """
    One-step oracle selective reuse.

    For each boundary row:
        drift > threshold
            -> refresh using current row

        drift <= threshold
            -> reuse previous row

    Both tensors:
        [batch, sequence_length, hidden_size]

    NOTE:
    This is an oracle/feasibility prototype.
    The current exact boundary has already been computed.
    Therefore, this function does NOT represent compute saving yet.
    """

    # --------------------------------
    # Row-wise normalized L2 drift
    # --------------------------------

    current_float = current_boundary.float()
    previous_float = previous_boundary.float()

    numerator = torch.linalg.vector_norm(
        current_float - previous_float,
        ord=2,
        dim=-1,
    )

    denominator = (
        torch.linalg.vector_norm(
            current_float,
            ord=2,
            dim=-1,
        )
        + eps
    )

    drift = numerator / denominator

    # Shape:
    # [batch, sequence_length]
    active_mask = drift > threshold

    # --------------------------------
    # Reuse previous rows
    # --------------------------------

    selective_boundary = (
        previous_boundary.clone()
    )

    # Replace active rows with
    # the exact current boundary.
    selective_boundary[
        active_mask
    ] = current_boundary[
        active_mask
    ]

    active_ratio = (
        active_mask
        .float()
        .mean()
        .item()
    )

    reuse_ratio = 1.0 - active_ratio

    return {
        "boundary": selective_boundary,
        "drift": drift,
        "active_mask": active_mask,
        "active_ratio": active_ratio,
        "reuse_ratio": reuse_ratio,
    }


@torch.no_grad()
def cloud_suffix_from_boundary(
    model,
    boundary_cloud,
    split_layer,
    cloud_rotary_emb,
    cloud_device,
):
    """
    Run only the cloud-side suffix from a supplied boundary.

    Uses the same cloud suffix structure as
    split_model.two_gpu_split_forward().
    """

    base_model = model.model

    hidden_states = boundary_cloud.to(
        cloud_device
    )

    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]

    # --------------------------------
    # Attention / positional info
    # --------------------------------

    attention_mask_cloud = torch.ones(
        (batch_size, seq_len),
        dtype=torch.bool,
        device=cloud_device,
    )

    cache_position_cloud = torch.arange(
        seq_len,
        device=cloud_device,
    )

    position_ids_cloud = (
        cache_position_cloud.unsqueeze(0)
    )

    position_embeddings_cloud = (
        cloud_rotary_emb(
            hidden_states,
            position_ids_cloud,
        )
    )

    # --------------------------------
    # Cloud suffix
    # --------------------------------

    for layer_idx in range(
        split_layer,
        len(base_model.layers),
    ):

        layer = base_model.layers[layer_idx]

        layer_output = layer(
            hidden_states,
            attention_mask=attention_mask_cloud,
            position_ids=position_ids_cloud,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position_cloud,
            position_embeddings=position_embeddings_cloud,
        )

        hidden_states = layer_output[0]

    # --------------------------------
    # Final norm + LM head
    # --------------------------------

    hidden_states = base_model.norm(
        hidden_states
    )

    logits = model.lm_head(
        hidden_states
    )

    return logits