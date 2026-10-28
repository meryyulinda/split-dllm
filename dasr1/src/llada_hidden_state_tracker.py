import torch


def normalized_row_drift(current, previous, eps=1e-8):
    """
    Normalized L2 drift per token row.

    current, previous:
        [batch, seq_len, hidden_dim]

    returns:
        [batch, seq_len]
    """
    current = current.float()
    previous = previous.float()

    numerator = torch.linalg.vector_norm(
        current - previous,
        ord=2,
        dim=-1,
    )

    denominator = (
        torch.linalg.vector_norm(
            current,
            ord=2,
            dim=-1,
        )
        + eps
    )

    return numerator / denominator


def _extract_hidden_tensor(output):
    """
    Robustly extract the hidden-state tensor from a block output.
    """

    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item

    raise TypeError(
        f"Cannot extract hidden tensor from block output type: "
        f"{type(output)}"
    )


class LLaDAHiddenStateTracker:
    """
    Capture hidden states from selected LLaDA transformer blocks.

    Expected LLaDA path:
        model.model.transformer.blocks[layer_idx]
    """

    def __init__(self, model, layer_indices):
        self.model = model
        self.layer_indices = list(layer_indices)

        self.hidden_history = {
            layer_idx: []
            for layer_idx in self.layer_indices
        }

        self.handles = []


    def _make_hook(self, layer_idx):

        def hook(module, inputs, output):

            hidden = _extract_hidden_tensor(output)

            # Store on CPU because this experiment is for
            # analysis, not latency measurement.
            self.hidden_history[layer_idx].append(
                hidden.detach().cpu().clone()
            )

        return hook


    def register(self):

        blocks = self.model.model.transformer.blocks

        for layer_idx in self.layer_indices:

            handle = blocks[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )

            self.handles.append(handle)


    def remove(self):

        for handle in self.handles:
            handle.remove()

        self.handles = []


    def num_states(self, layer_idx):

        return len(
            self.hidden_history[layer_idx]
        )