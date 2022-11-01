import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import time

from editable_model import EditableModel
from utils import _last_encoder_state, _logits

class LU(EditableModel):
    """
    Representation lookup approach. Does not require training.
    """

    def __init__(self, model, config, model_constructor, memory=None):
        super().__init__(model, config, model_constructor)

        self.memory = memory

    def forward(self, *inputs, **kwargs):
        if "bert" in self.config.model.name.lower():
            output, encoder_states = self.model(*inputs, **kwargs, output_hidden_states=True)
        else:
            model_output = self.model(*inputs, **kwargs, output_hidden_states=True)
            encoder_states = _last_encoder_state(model_output)
            output = _logits(model_output)

        if self.memory is not None:
            for i, encoder_state in enumerate(encoder_states):
                if "gpt2" in self.config.model.name.lower():
                    # NOTE: broken
                    memory_prefixes, memory_labels = self.memory
                    prefix_means = encoder_state.cumsum(0).detach() / torch.arange(1, encoder_state.shape[0] + 1, device=encoder_state.device).view(-1, 1)
                    dist_mat = (prefix_means.unsqueeze(1) - memory_prefixes.unsqueeze(0)).norm(2, dim=-1)

                    min_dists, min_idxs = dist_mat.min(-1)
                    memory_mask = (min_dists < self.config.lu.threshold)
                    onehot_logits = self.config.lu.onehot_logit * F.one_hot(memory_labels[min_idxs], output.shape[-1]).float()
                    output[i, memory_mask] = onehot_logits[memory_mask]
                elif "bart" in self.config.model.name.lower() or "t5" in self.config.model.name.lower():
                    avg_encoder_state = encoder_state.detach().mean(0)
                    memory_keys, memory_labels = self.memory
                    dists = torch.norm(avg_encoder_state - memory_keys, dim=-1)
                    closest_dist = dists.min()
                    closest_idx = dists.argmin()
                    closest_v = memory_labels[closest_idx]

                    if closest_dist < self.config.lu.threshold:
                        output[i] = torch.zeros((1, kwargs['labels'].shape[1], output.shape[2]), device=output.device)
                        for j, idx in enumerate(closest_v):
                            if j >= output.shape[1]:
                                break
                            output[i, j, idx] = self.config.lu.onehot_logit
                        if "t5" not in self.config.model.name.lower():
                            # T5 does not shift targets in the loss
                            output[i] = output[i].roll(-1, -2)
                else:
                    avg_encoder_state = encoder_state.detach().mean(0)
                    memory_keys, memory_labels = self.memory
                    dists = torch.norm(avg_encoder_state - memory_keys, dim=-1)
                    closest_dist = dists.min()
                    closest_idx = dists.argmin()
                    closest_v = memory_labels[closest_idx]

                    if closest_dist < self.config.lu.threshold:
                        output[i] = self.config.lu.onehot_logit * (2 * closest_v - 1)  # Return onehot_logit or -onehot_logit

        return output

    def edit(self, batch, condition=None):
        edit_model = self.model.eval()
        if "bert" in self.config.model.name.lower():
            _, encoder_states = self.model(**batch, output_hidden_states=True)
        else:
            encoder_states = _last_encoder_state(self.model(**batch, output_hidden_states=True))

        memory_keys = []
        memory_labels = []
        for encoder_state, label in zip(encoder_states, batch["labels"]):
            if "gpt2" in self.config.model.name.lower():
                # NOTE: broken
                avg_encoder_states = (encoder_state.cumsum(0).detach() / torch.arange(1, encoder_state.shape[0] + 1, device=encoder_state.device).view(-1, 1))[-10:, :]
                memory = (avg_encoder_states, label[-10:])
            else:
                avg_encoder_state = encoder_state.detach().mean(0)
                memory_keys.append(avg_encoder_state)
                memory_labels.append(label)

        memory = (torch.stack(memory_keys), torch.stack(memory_labels))
        return LU(self.model.eval(), self.config, self.model_constructor, memory), {}
