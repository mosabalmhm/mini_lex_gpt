from __future__ import annotations

import torch
from torch import nn

from mini_lex_gpt.training.loss import (
    next_token_cross_entropy,
)

def train_step(
    model: nn.Modele,
    optimizer: torch.optim.Optimizeer,
    token_ids: torch.Tensor,
    max_gradient_norm: float,
) -> tuple[torch.Tensor, torch.Tensor]:

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(
        token_ids
    )

    loss = next_token_cross_entropy(
        logits=logits,
        token_ids=token_ids,
    )

    loss.backward()

    gradient_norm = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_gradient_norm,
            norm_type=2.0,
            error_if_nonfinite=True,
        )
    )

    optimizer.step()

    return loss.detach(), gradient_norm.detach()
