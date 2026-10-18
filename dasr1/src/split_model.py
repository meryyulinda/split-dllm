import torch

@torch.no_grad()
def manual_split_forward(model, input_ids, attention_mask, split_layer, profile=False,):
    """
    Simulate a logical edge-cloud model split on the SAME GPU.

    Layer notation
    --------------
    `split_layer` uses human-readable 1-based numbering.

    Example:
        split_layer = 12

        Edge / Stage 1:
            embedding + position setup
            + Transformer layers 1..12
            (Python indices 0..11)

        Cloud / Stage 2:
            Transformer layers 13..28
            (Python indices 12..27)
            + final normalization
            + LM head

    The hidden state after `split_layer` becomes the logical
    edge -> cloud boundary tensor.

    When profile=True, CUDA events measure Stage 1 and Stage 2
    compute latency separately. No real network transfer is
    performed in this function.
    """

    base_model = model.model

    # -------------------------------------------------
    # Optional CUDA profiling setup
    # -------------------------------------------------

    if profile:
        stage1_start = torch.cuda.Event(
            enable_timing=True
        )
        stage1_end = torch.cuda.Event(
            enable_timing=True
        )

        stage2_start = torch.cuda.Event(
            enable_timing=True
        )
        stage2_end = torch.cuda.Event(
            enable_timing=True
        )

        # Stage 1 includes embedding, position setup,
        # and all prefix Transformer layers.
        stage1_start.record()


    # -------------------------------------------------
    # STAGE 1 / EDGE
    #
    # A. Input embedding
    # -------------------------------------------------

    hidden_states = base_model.embed_tokens(input_ids)

    seq_len = hidden_states.shape[1]

    # --------------------------------
    # B. Position information
    # --------------------------------

    cache_position = torch.arange(seq_len, device=hidden_states.device,)

    position_ids = (cache_position.unsqueeze(0))

    position_embeddings = (
        base_model.rotary_emb(hidden_states, position_ids,)
    )

    # --------------------------------
    # C. EDGE PREFIX
    #
    # Human-readable:
    #     layers 1 .. split_layer
    #
    # Python indices:
    #     0 .. split_layer - 1
    # --------------------------------

    for layer_idx in range(split_layer):

        layer = base_model.layers[layer_idx]

        layer_output = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        hidden_states = (layer_output[0])

    # Stage 1 computation finishes here,
    # immediately after the final edge layer.
    if profile:
        stage1_end.record()

    # -------------------------------------------------
    # Logical edge -> cloud boundary
    #
    # Clone only records/materializes the boundary
    # tensor for analysis. Its copy cost is intentionally
    # excluded from Stage 1 and Stage 2 compute latency.
    # -------------------------------------------------

    boundary = hidden_states.clone()

    if profile:
        stage1_end.record()
        stage2_start.record()

    # -------------------------------------------------
    # STAGE 2 / CLOUD
    # -------------------------------------------------

    if profile:
        stage2_start.record()

    # -------------------------------------------------
    # D. Cloud suffix
    #
    # Human-readable:
    #     layers split_layer + 1 .. final layer
    #
    # Example split_layer=12:
    #     layers 13..28
    #
    # Python indices:
    #     split_layer .. len(layers)-1
    # -------------------------------------------------

    for layer_idx in range(
        split_layer,
        len(base_model.layers),
    ):

        layer = base_model.layers[layer_idx]

        layer_output = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        hidden_states = (layer_output[0])

    # --------------------------------
    # E. Final normalization + LM head
    # --------------------------------

    hidden_states = (base_model.norm(hidden_states))

    logits = model.lm_head(hidden_states)

    # -------------------------------------------------
    # Finish Stage 2 profiling
    # -------------------------------------------------

    if profile:
        stage2_end.record()

        torch.cuda.synchronize()

        stage1_ms = stage1_start.elapsed_time(
            stage1_end
        )

        stage2_ms = stage2_start.elapsed_time(
            stage2_end
        )

    else:
        stage1_ms = None
        stage2_ms = None

    # -------------------------------------------------
    # Output
    # -------------------------------------------------

    return {
        "logits": logits,
        "boundary": boundary,
        "stage1_ms": stage1_ms,
        "stage2_ms": stage2_ms,
    }