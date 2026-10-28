import torch
import time
import copy

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

def configure_two_gpu_split(
    model,
    split_layer,
    edge_device="cuda:0",
    cloud_device="cuda:1",
):
    """
    Physically partition Dream across two GPUs.

    Human-readable split numbering:
        split_layer = 12

    Edge / GPU 0:
        embedding
        Transformer layers 1..12

    Cloud / GPU 1:
        Transformer layers 13..28
        final norm
        LM head

    Module placement is performed outside timed inference.
    """

    base_model = model.model

    # ---------------------------------------------
    # Edge-side modules
    # ---------------------------------------------

    base_model.embed_tokens.to(edge_device)
    base_model.rotary_emb.to(edge_device)

    for layer_idx, layer in enumerate(
        base_model.layers
    ):
        if layer_idx < split_layer:
            layer.to(edge_device)
        else:
            layer.to(cloud_device)

    # ---------------------------------------------
    # Cloud-side modules
    # ---------------------------------------------

    base_model.norm.to(cloud_device)
    model.lm_head.to(cloud_device)

    # Separate lightweight RoPE module for cloud.
    # This avoids transmitting position embeddings
    # across GPUs with every diffusion state.
    cloud_rotary_emb = copy.deepcopy(
        base_model.rotary_emb
    ).to(cloud_device)

    torch.cuda.synchronize()

    return cloud_rotary_emb


@torch.no_grad()
def two_gpu_split_forward(
    model,
    input_ids,
    split_layer,
    cloud_rotary_emb,
    edge_device="cuda:0",
    cloud_device="cuda:1",
    profile=False,
):
    """
    Execute a physical edge-cloud split across two GPUs.

    Layer numbering is human-readable:
        split_layer = 12

    Stage 1 / Edge GPU:
        token embedding
        + positional information
        + Transformer layers 1..12

    Transfer:
        boundary hidden state from GPU 0 -> GPU 1

    Stage 2 / Cloud GPU:
        positional information reconstructed locally
        + Transformer layers 13..28
        + final normalization
        + LM head

    Only the boundary hidden state is physically transmitted
    between GPUs. Position information is reconstructed locally.

    Profiling returns:
        stage1_ms
        transfer_ms
        stage2_ms
        component_sum_ms
        measured_e2e_ms
    """

    base_model = model.model

    # -------------------------------------------------
    # Prepare input on edge GPU
    # -------------------------------------------------

    input_ids = input_ids.to(
        edge_device
    )

    # -------------------------------------------------
    # Synchronize before end-to-end timing
    # -------------------------------------------------

    if profile:
        torch.cuda.synchronize(
            edge_device
        )
        torch.cuda.synchronize(
            cloud_device
        )

        e2e_start = time.perf_counter()

    # =================================================
    # STAGE 1 / EDGE GPU
    # =================================================

    if profile:

        with torch.cuda.device(
            edge_device
        ):
            stage1_start = torch.cuda.Event(
                enable_timing=True
            )
            stage1_end = torch.cuda.Event(
                enable_timing=True
            )

            stage1_start.record()

    # -------------------------------------------------
    # A. Token embedding
    # -------------------------------------------------

    hidden_states = (
        base_model.embed_tokens(
            input_ids
        )
    )

    seq_len = hidden_states.shape[1]

    # -------------------------------------------------
    # B. Edge-side attention / position information
    # -------------------------------------------------

    attention_mask_edge = torch.ones_like(
        input_ids,
        dtype=torch.bool,
        device=edge_device,
    )

    cache_position_edge = torch.arange(
        seq_len,
        device=edge_device,
    )

    position_ids_edge = (
        cache_position_edge.unsqueeze(0)
    )

    position_embeddings_edge = (
        base_model.rotary_emb(
            hidden_states,
            position_ids_edge,
        )
    )

    # -------------------------------------------------
    # C. Edge prefix
    #
    # Human-readable:
    #     Layers 1 .. split_layer
    #
    # Python:
    #     indices 0 .. split_layer - 1
    # -------------------------------------------------

    for layer_idx in range(
        split_layer
    ):

        layer = (
            base_model.layers[
                layer_idx
            ]
        )

        layer_output = layer(
            hidden_states,
            attention_mask=
                attention_mask_edge,
            position_ids=
                position_ids_edge,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=
                cache_position_edge,
            position_embeddings=
                position_embeddings_edge,
        )

        hidden_states = (
            layer_output[0]
        )

    # -------------------------------------------------
    # Finish Stage 1 timing
    # -------------------------------------------------

    if profile:

        with torch.cuda.device(
            edge_device
        ):
            stage1_end.record()

        torch.cuda.synchronize(
            edge_device
        )

        stage1_ms = (
            stage1_start.elapsed_time(
                stage1_end
            )
        )

    else:
        stage1_ms = None

    # =================================================
    # EDGE -> CLOUD BOUNDARY
    # =================================================

    boundary_edge = hidden_states

    boundary_bytes = (
        boundary_edge.numel()
        * boundary_edge.element_size()
    )

    # =================================================
    # PHYSICAL GPU 0 -> GPU 1 TRANSFER
    # =================================================

    if profile:

        # Ensure previous edge compute has completed
        # before starting transfer timing.
        torch.cuda.synchronize(
            edge_device
        )

        torch.cuda.synchronize(
            cloud_device
        )

        transfer_start = (
            time.perf_counter()
        )

    boundary_cloud = (
        boundary_edge.to(
            cloud_device
        )
    )

    if profile:

        # .to() may be asynchronous depending on
        # the execution path, so synchronize GPU 1
        # before stopping the timer.
        torch.cuda.synchronize(
            cloud_device
        )

        transfer_ms = (
            time.perf_counter()
            - transfer_start
        ) * 1000

    else:
        transfer_ms = None

    # =================================================
    # STAGE 2 / CLOUD GPU
    # =================================================

    if profile:

        with torch.cuda.device(
            cloud_device
        ):
            stage2_start = torch.cuda.Event(
                enable_timing=True
            )
            stage2_end = torch.cuda.Event(
                enable_timing=True
            )

            stage2_start.record()

    hidden_states = boundary_cloud

    # -------------------------------------------------
    # D. Reconstruct mask / position information
    #    locally on cloud GPU
    # -------------------------------------------------

    attention_mask_cloud = torch.ones(
        (
            input_ids.shape[0],
            seq_len,
        ),
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

    # -------------------------------------------------
    # E. Cloud suffix
    #
    # Human-readable:
    #     Layers split_layer+1 .. 28
    #
    # Example split_layer=12:
    #     Layers 13..28
    #
    # Python:
    #     indices split_layer .. 27
    # -------------------------------------------------

    for layer_idx in range(
        split_layer,
        len(base_model.layers),
    ):

        layer = (
            base_model.layers[
                layer_idx
            ]
        )

        layer_output = layer(
            hidden_states,
            attention_mask=
                attention_mask_cloud,
            position_ids=
                position_ids_cloud,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=
                cache_position_cloud,
            position_embeddings=
                position_embeddings_cloud,
        )

        hidden_states = (
            layer_output[0]
        )

    # -------------------------------------------------
    # F. Final normalization + LM head
    # -------------------------------------------------

    hidden_states = (
        base_model.norm(
            hidden_states
        )
    )

    logits = model.lm_head(
        hidden_states
    )

    # -------------------------------------------------
    # Finish Stage 2 timing
    # -------------------------------------------------

    if profile:

        with torch.cuda.device(
            cloud_device
        ):
            stage2_end.record()

        torch.cuda.synchronize(
            cloud_device
        )

        stage2_ms = (
            stage2_start.elapsed_time(
                stage2_end
            )
        )

    else:
        stage2_ms = None

    # =================================================
    # END-TO-END TIMING
    # =================================================

    if profile:

        torch.cuda.synchronize(
            cloud_device
        )

        measured_e2e_ms = (
            time.perf_counter()
            - e2e_start
        ) * 1000

        component_sum_ms = (
            stage1_ms
            + transfer_ms
            + stage2_ms
        )

    else:
        measured_e2e_ms = None
        component_sum_ms = None

    # =================================================
    # OUTPUT
    # =================================================

    return {
        "logits":
            logits,

        "boundary":
            boundary_cloud,

        "boundary_bytes":
            boundary_bytes,

        "stage1_ms":
            stage1_ms,

        "transfer_ms":
            transfer_ms,

        "stage2_ms":
            stage2_ms,

        "component_sum_ms":
            component_sum_ms,

        "measured_e2e_ms":
            measured_e2e_ms,
    }

@torch.no_grad()
def two_gpu_delta_split_forward(
    model,
    input_ids,
    split_layer,
    cloud_rotary_emb,
    prev_boundary_edge=None,
    prev_boundary_cloud=None,
    edge_device="cuda:0",
    cloud_device="cuda:1",
    profile=False,
):
    """
    Physical two-GPU split with exact synchronized delta transmission.

    First diffusion state:
        send the full boundary tensor.

    Subsequent states:
        compare current edge boundary against previous edge boundary,
        identify rows that changed exactly,
        send only:
            1. changed row indices
            2. changed row values

        Cloud reconstructs the exact current boundary from its
        synchronized previous copy.

    No approximation threshold is used in this experiment.
    """

    base_model = model.model

    input_ids = input_ids.to(
        edge_device
    )

    # =================================================
    # END-TO-END TIMER
    # =================================================

    if profile:
        torch.cuda.synchronize(edge_device)
        torch.cuda.synchronize(cloud_device)

        e2e_start = time.perf_counter()

    # =================================================
    # STAGE 1 / EDGE
    # =================================================

    if profile:

        with torch.cuda.device(edge_device):

            stage1_start = torch.cuda.Event(
                enable_timing=True
            )

            stage1_end = torch.cuda.Event(
                enable_timing=True
            )

            stage1_start.record()

    hidden_states = base_model.embed_tokens(
        input_ids
    )

    seq_len = hidden_states.shape[1]

    attention_mask_edge = torch.ones_like(
        input_ids,
        dtype=torch.bool,
        device=edge_device,
    )

    cache_position_edge = torch.arange(
        seq_len,
        device=edge_device,
    )

    position_ids_edge = (
        cache_position_edge.unsqueeze(0)
    )

    position_embeddings_edge = (
        base_model.rotary_emb(
            hidden_states,
            position_ids_edge,
        )
    )

    for layer_idx in range(split_layer):

        layer = base_model.layers[
            layer_idx
        ]

        layer_output = layer(
            hidden_states,
            attention_mask=attention_mask_edge,
            position_ids=position_ids_edge,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position_edge,
            position_embeddings=position_embeddings_edge,
        )

        hidden_states = layer_output[0]

    if profile:

        with torch.cuda.device(
            edge_device
        ):
            stage1_end.record()

        torch.cuda.synchronize(
            edge_device
        )

        stage1_ms = (
            stage1_start.elapsed_time(
                stage1_end
            )
        )

    else:
        stage1_ms = None

    boundary_edge = hidden_states

    full_boundary_bytes = (
        boundary_edge.numel()
        * boundary_edge.element_size()
    )

    # =================================================
    # EXACT DELTA ENCODING
    # =================================================

    if profile:
        torch.cuda.synchronize(
            edge_device
        )

        encode_start = (
            time.perf_counter()
        )

    # First state:
    # no synchronized previous boundary exists.
    if prev_boundary_edge is None:

        is_initial_sync = True

        changed_indices_edge = torch.arange(
            boundary_edge.shape[1],
            device=edge_device,
            dtype=torch.long,
        )

        changed_rows_edge = (
            boundary_edge[0]
        )

    else:

        is_initial_sync = False

        # Shape:
        # boundary = [1, sequence_length, hidden_dim]
        #
        # A row is considered changed if ANY hidden
        # dimension differs exactly.
        changed_mask = torch.any(
            boundary_edge != prev_boundary_edge,
            dim=-1,
        )[0]

        changed_indices_edge = (
            torch.nonzero(
                changed_mask,
                as_tuple=False,
            )
            .squeeze(-1)
        )

        changed_rows_edge = (
            boundary_edge[
                0,
                changed_indices_edge,
                :
            ]
        )

    if profile:

        torch.cuda.synchronize(
            edge_device
        )

        encode_ms = (
            time.perf_counter()
            - encode_start
        ) * 1000

    else:
        encode_ms = None

    changed_rows = (
        changed_indices_edge.numel()
    )

    total_rows = boundary_edge.shape[1]

    changed_row_ratio = (
        changed_rows / total_rows
        if total_rows > 0
        else 0.0
    )

    # Actual transmitted payload:
    # row values + int64 row indices.
    value_bytes = (
        changed_rows_edge.numel()
        * changed_rows_edge.element_size()
    )

    index_bytes = (
        changed_indices_edge.numel()
        * changed_indices_edge.element_size()
    )

    delta_bytes = (
        value_bytes
        + index_bytes
    )

    # =================================================
    # PHYSICAL DELTA TRANSFER
    # GPU EDGE -> GPU CLOUD
    # =================================================

    if profile:

        torch.cuda.synchronize(
            edge_device
        )

        torch.cuda.synchronize(
            cloud_device
        )

        transfer_start = (
            time.perf_counter()
        )

    changed_indices_cloud = (
        changed_indices_edge.to(
            cloud_device
        )
    )

    changed_rows_cloud = (
        changed_rows_edge.to(
            cloud_device
        )
    )

    if profile:

        torch.cuda.synchronize(
            cloud_device
        )

        delta_transfer_ms = (
            time.perf_counter()
            - transfer_start
        ) * 1000

    else:
        delta_transfer_ms = None

    # =================================================
    # CLOUD-SIDE EXACT RECONSTRUCTION
    # =================================================

    if profile:

        torch.cuda.synchronize(
            cloud_device
        )

        reconstruction_start = (
            time.perf_counter()
        )

    if prev_boundary_cloud is None:

        # Initial synchronization.
        boundary_cloud = (
            changed_rows_cloud
            .unsqueeze(0)
        )

    else:

        # Start from the synchronized previous state.
        boundary_cloud = (
            prev_boundary_cloud.clone()
        )

        # Replace only rows received from edge.
        if changed_indices_cloud.numel() > 0:

            boundary_cloud[
                0,
                changed_indices_cloud,
                :
            ] = changed_rows_cloud

    if profile:

        torch.cuda.synchronize(
            cloud_device
        )

        reconstruction_ms = (
            time.perf_counter()
            - reconstruction_start
        ) * 1000

    else:
        reconstruction_ms = None

    # =================================================
    # STAGE 2 / CLOUD
    # =================================================

    if profile:

        with torch.cuda.device(
            cloud_device
        ):

            stage2_start = torch.cuda.Event(
                enable_timing=True
            )

            stage2_end = torch.cuda.Event(
                enable_timing=True
            )

            stage2_start.record()

    hidden_states = boundary_cloud

    attention_mask_cloud = torch.ones(
        (
            input_ids.shape[0],
            seq_len,
        ),
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

    for layer_idx in range(
        split_layer,
        len(base_model.layers),
    ):

        layer = base_model.layers[
            layer_idx
        ]

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

    hidden_states = (
        base_model.norm(
            hidden_states
        )
    )

    logits = model.lm_head(
        hidden_states
    )

    if profile:

        with torch.cuda.device(
            cloud_device
        ):
            stage2_end.record()

        torch.cuda.synchronize(
            cloud_device
        )

        stage2_ms = (
            stage2_start.elapsed_time(
                stage2_end
            )
        )

    else:
        stage2_ms = None

    # =================================================
    # END-TO-END
    # =================================================

    if profile:

        torch.cuda.synchronize(
            cloud_device
        )

        measured_e2e_ms = (
            time.perf_counter()
            - e2e_start
        ) * 1000

        component_sum_ms = (
            stage1_ms
            + encode_ms
            + delta_transfer_ms
            + reconstruction_ms
            + stage2_ms
        )

    else:
        measured_e2e_ms = None
        component_sum_ms = None

    return {
        "logits":
            logits,

        # Keep synchronized state for next step.
        "boundary_edge":
            boundary_edge.detach(),

        "boundary_cloud":
            boundary_cloud.detach(),

        "full_boundary_bytes":
            full_boundary_bytes,

        "delta_bytes":
            delta_bytes,

        "value_bytes":
            value_bytes,

        "index_bytes":
            index_bytes,

        "changed_rows":
            changed_rows,

        "total_rows":
            total_rows,

        "changed_row_ratio":
            changed_row_ratio,

        "initial_sync":
            is_initial_sync,

        "stage1_ms":
            stage1_ms,

        "encode_ms":
            encode_ms,

        "transfer_ms":
            delta_transfer_ms,

        "reconstruction_ms":
            reconstruction_ms,

        "stage2_ms":
            stage2_ms,

        "component_sum_ms":
            component_sum_ms,

        "measured_e2e_ms":
            measured_e2e_ms,
    }