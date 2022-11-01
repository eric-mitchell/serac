import torch.nn as nn

from losses import masked_log_probs
from utils import _logits, shift_targets


class EditableModel(nn.Module):
    def __init__(self, model, config, model_constructor):
        super().__init__()

        self.model = model
        self.config = config
        self.model_constructor = model_constructor

        def _edit_loss_fn(pred, targ, **kwargs):
            return masked_log_probs(pred, targ, shift=shift_targets(self.config), **kwargs)
        self.edit_loss_fn = _edit_loss_fn
        self.loc_loss_fn = _edit_loss_fn

    def edit(self, batch, condition=None, detach_history=False):
        raise NotImplementedError

    def forward(self, *inputs, **kwargs):
        return _logits(self.model(*inputs, **kwargs))

    def outer_parameters(self, grouped=False):
        if grouped:
            return [dict(params=self.parameters(), lr=self.config.lr)]
        else:
            return list(self.parameters())

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def base_loss(self, input_ids, attention_masks, label_ids):
        pass
