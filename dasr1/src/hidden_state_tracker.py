import torch

class HiddenDriftTracker:
    def __init__(self, selected_layers, prompt_length=0, eps=1e-8,):
        self.selected_layers = selected_layers
        self.prompt_length = prompt_length
        self.eps = eps

        # Previous hidden state for each layer
        self.previous = {}

        # Results that will eventually become a DataFrame
        self.records = []

        # Hook handles, needed so we can remove hooks later
        self.handles = []

        # Number of forward calls seen at each layer
        self.call_counts = {layer: 0 for layer in selected_layers}

    def _normalized_drift(self, current, previous,):
        """
        Compute row-wise normalized L2 drift.

        current/previous shape:
            [sequence_length, hidden_size]
        
        return shape:
            [sequence_length]
        """

        numerator = torch.linalg.vector_norm(current - previous, ord=2, dim=-1)
        denominator = torch.linalg.vector_norm(current, ord=2, dim=-1) + self.eps
        return numerator / denominator

    def _create_hook(self, layer_number):

        def hook(module, input, output):

            # DreamDecoderLayer returns a tuple.
            # output[0] is the hidden state.
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # batch size = 1
            hidden = hidden[0]

            # Only analyze generated region,
            # not the fixed prompt.
            if self.prompt_length > 0:
                hidden = hidden[self.prompt_length:]

            # Detach from computation graph.
            #
            # Convert to float32 for more stable distance calculations.
            current = (hidden.detach().float().cpu())

            call_index = self.call_counts[layer_number]

            # # TEMPORARY DEBUG to check whether forward calls equal to denoising steps.
            # if layer_number == 4:
            #     print(
            #         f"Layer {layer_number} | "
            #         f"forward call {call_index} | "
            #         f"shape {current.shape}"
            #     )

            if layer_number in self.previous:

                previous = self.previous[layer_number]

                drift = self._normalized_drift(current, previous,)
                
                thresholds = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2,]

                active_ratios = {}

                for threshold in thresholds:

                    active_ratio = ((drift > threshold).float().mean().item())

                    active_ratios[threshold] = active_ratio

                self.records.append({
                    "layer": layer_number,
                    "transition": call_index,
                    "mean_drift": (drift.mean().item()),
                    "median_drift": (drift.median().item()),
                    "max_drift": (drift.max().item()),
                    "min_drift": (drift.min().item()),
                    "active_ratio_1e-4": (active_ratios[1e-4]),
                    "active_ratio_5e-4": (active_ratios[5e-4]),
                    "active_ratio_1e-3": (active_ratios[1e-3]),
                    "active_ratio_5e-3": (active_ratios[5e-3]),
                    "active_ratio_1e-2": (active_ratios[1e-2]),
                })

            # Save current state for the next denoising iteration.
            self.previous[layer_number] = current

            self.call_counts[layer_number] += 1

        return hook

    def register(self, model):
        """
        Attach hooks to selected Dream Transformer layers.
        """

        for layer_number in self.selected_layers:

            layer_index = layer_number - 1

            layer = model.model.layers[layer_index]

            handle = layer.register_forward_hook(self._create_hook(layer_number))

            self.handles.append(handle)
    
    def remove(self):
        """
        Remove all hooks after generation.
        """

        for handle in self.handles:
            handle.remove()

        self.handles = []