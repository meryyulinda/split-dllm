import torch

@torch.no_grad()
def manual_split_forward(model, input_ids, attention_mask, split_layer,):
    """
    Logical split on the SAME GPU.

    split_layer uses human-readable numbering:
    split_layer=12 means:
        edge  = layers 1..12
        cloud = layers 13..28
    """

    base_model = model.model

    # --------------------------------
    # 1. Input embedding
    # --------------------------------

    hidden_states = base_model.embed_tokens(input_ids)

    seq_len = hidden_states.shape[1]

    # --------------------------------
    # 2. Position information
    # --------------------------------

    cache_position = torch.arange(seq_len, device=hidden_states.device,)

    position_ids = (cache_position.unsqueeze(0))

    position_embeddings = (
        base_model.rotary_emb(hidden_states, position_ids,)
    )

    # --------------------------------
    # 3. EDGE PREFIX
    # layers 0 .. split_layer-1
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

    # This is our simulated
    # edge -> cloud boundary tensor
    boundary = hidden_states.clone()

    # --------------------------------
    # 4. CLOUD SUFFIX
    # --------------------------------

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
    # 5. Final norm + LM head
    # --------------------------------

    hidden_states = (base_model.norm(hidden_states))

    logits = model.lm_head(hidden_states)

    return {
        "logits": logits,
        "boundary": boundary,
    }




