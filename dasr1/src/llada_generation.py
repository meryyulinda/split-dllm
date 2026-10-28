import torch
import numpy as np
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    """
    Same behavior as official LLaDA generate.py.
    """
    if temperature == 0:
        return logits

    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature

    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute how many masked tokens are transferred
    at each denoising step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0),
            steps,
            device=mask_index.device,
            dtype=torch.int64,
        )
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate_with_history(
    model,
    prompt,
    attention_mask=None,
    steps=64,
    gen_length=64,
    block_length=64,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    """
    LLaDA generation with token-state trajectory capture.

    Returns
    -------
    x : torch.Tensor
        Final generated sequence.

    token_history : list[torch.Tensor]
        Sequence states including:
            state 0  = initial masked state
            state 1  = after denoising update 1
            ...
            state N  = after final denoising update

        With:
            steps=64
            gen_length=64
            block_length=64

        expected:
            65 states
            64 transitions
    """

    # ---------------------------------------------------------
    # 1. Construct initial masked sequence
    # ---------------------------------------------------------

    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )

    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (prompt.shape[0], gen_length),
                    dtype=attention_mask.dtype,
                    device=model.device,
                ),
            ],
            dim=-1,
        )

    prompt_index = x != mask_id

    # ---------------------------------------------------------
    # IMPORTANT:
    # Store INITIAL pre-denoising state.
    # ---------------------------------------------------------

    token_history = [
        x.detach().clone()
    ]

    # Optional metadata per transition
    transition_history = []

    # ---------------------------------------------------------
    # 2. Block setup
    # ---------------------------------------------------------

    assert gen_length % block_length == 0

    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0

    steps_per_block = steps // num_blocks

    # ---------------------------------------------------------
    # 3. Denoising
    # ---------------------------------------------------------

    global_step = 0

    for num_block in range(num_blocks):

        block_start = (
            prompt.shape[1]
            + num_block * block_length
        )

        block_end = (
            prompt.shape[1]
            + (num_block + 1) * block_length
        )

        block_mask_index = (
            x[:, block_start:block_end]
            == mask_id
        )

        num_transfer_tokens = get_num_transfer_tokens(
            block_mask_index,
            steps_per_block,
        )

        for i in range(steps_per_block):

            # ---------------------------------------------
            # Current masked positions
            # ---------------------------------------------

            mask_index = x == mask_id

            candidate_mask_index = mask_index.clone()

            candidate_mask_index[:, :block_start] = False
            candidate_mask_index[:, block_end:] = False

            # ---------------------------------------------
            # Model forward
            # ---------------------------------------------

            if cfg_scale > 0.0:

                un_x = x.clone()
                un_x[prompt_index] = mask_id

                x_ = torch.cat(
                    [x, un_x],
                    dim=0,
                )

                if attention_mask is not None:
                    attention_mask_ = torch.cat(
                        [attention_mask, attention_mask],
                        dim=0,
                    )
                else:
                    attention_mask_ = None

                logits = model(
                    x_,
                    attention_mask=attention_mask_,
                ).logits

                logits, un_logits = torch.chunk(
                    logits,
                    2,
                    dim=0,
                )

                logits = (
                    un_logits
                    + (cfg_scale + 1)
                    * (logits - un_logits)
                )

            else:

                logits = model(
                    x,
                    attention_mask=attention_mask,
                ).logits

            # ---------------------------------------------
            # Predict clean token x0
            # ---------------------------------------------

            logits_with_noise = add_gumbel_noise(
                logits,
                temperature=temperature,
            )

            x0 = torch.argmax(
                logits_with_noise,
                dim=-1,
            )

            # ---------------------------------------------
            # Confidence used for token transfer
            # ---------------------------------------------

            if remasking == "low_confidence":

                p = F.softmax(
                    logits,
                    dim=-1,
                )

                x0_p = torch.squeeze(
                    torch.gather(
                        p,
                        dim=-1,
                        index=torch.unsqueeze(x0, -1),
                    ),
                    -1,
                )

            elif remasking == "random":

                x0_p = torch.rand(
                    (x0.shape[0], x0.shape[1]),
                    device=x0.device,
                )

            else:
                raise NotImplementedError(
                    f"Unknown remasking strategy: {remasking}"
                )

            # Future blocks cannot be transferred yet.
            x0_p[:, block_end:] = -np.inf

            # Keep already-unmasked tokens unchanged.
            x0 = torch.where(
                mask_index,
                x0,
                x,
            )

            confidence = torch.where(
                mask_index,
                x0_p,
                -np.inf,
            )

            # ---------------------------------------------
            # Select positions to reveal
            # ---------------------------------------------

            transfer_index = torch.zeros_like(
                x0,
                dtype=torch.bool,
                device=x0.device,
            )

            for j in range(confidence.shape[0]):

                _, select_index = torch.topk(
                    confidence[j],
                    k=num_transfer_tokens[j, i],
                )

                transfer_index[
                    j,
                    select_index
                ] = True

            # ---------------------------------------------
            # THE ACTUAL DENOISING UPDATE
            # ---------------------------------------------

            previous_x = x.detach().clone()

            x[transfer_index] = x0[transfer_index]

            # ---------------------------------------------
            # Store NEW state AFTER update.
            #
            # initial state already exists in history,
            # therefore every denoising update adds
            # exactly one new state.
            # ---------------------------------------------

            token_history.append(
                x.detach().clone()
            )

            # ---------------------------------------------
            # Transition metadata
            # ---------------------------------------------

            changed_mask = (
                previous_x != x
            )

            generated_region = slice(
                prompt.shape[1],
                prompt.shape[1] + gen_length,
            )

            generated_changed_mask = (
                changed_mask[:, generated_region]
            )

            changed_tokens = (
                generated_changed_mask
                .sum()
                .item()
            )

            total_generated_tokens = (
                generated_changed_mask.numel()
            )

            changed_ratio = (
                changed_tokens
                / total_generated_tokens
            )

            transition_history.append(
                {
                    "transition": global_step + 1,
                    "block": num_block,
                    "step_in_block": i,
                    "changed_tokens": changed_tokens,
                    "changed_ratio": changed_ratio,
                }
            )

            global_step += 1

    # ---------------------------------------------------------
    # 4. Sanity checks
    # ---------------------------------------------------------

    expected_transitions = steps
    expected_states = steps + 1

    assert len(transition_history) == expected_transitions, (
        f"Expected {expected_transitions} transitions, "
        f"got {len(transition_history)}"
    )

    assert len(token_history) == expected_states, (
        f"Expected {expected_states} states, "
        f"got {len(token_history)}"
    )

    return x, token_history, transition_history