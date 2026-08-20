from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from mini_lex_gpt.training.loss import (
    next_token_cross_entropy,
)

@torch.no_grad()
def evaluate_language_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:

    model.eval()
    
    weighted_loss_sum = 0.0
    prediction_count = 0

    for token_ids in dataloader:
        token_ids = token_ids.to(
            device
        )

        logits = model(
            token_ids
        )

        loss = next_token_cross_entropy(
            logits=logits,
            token_ids=token_ids,
        )

        batch_prediction_count = (
           token_ids.size(0)
           * (
               token_ids.size(1)
               - 1
           )
        )
        
        weighted_loss_sum += (
            loss.item()
            * batch_prediction_count
        )

        prediction_count += (
            batch_prediction_count
        )

    if prediction_count == 0:
        raise RuntimeError(
            "Evaluation data set is empty."

    return (
        weighted_loss_sum
        / prediction_count
    )
