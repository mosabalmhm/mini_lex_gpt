from __future__ import annotations

import torch
import torch.nn.functional as F

def next_token_cross_entropy(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """

    Compute autoregressive nect-token cross-entropy loss.

    Expected shapes:
        logits:    [batch_size, sequence_length, vocab_size]
        token_ids: [batch_size, sequence_length]

    Returns:
        Scaler loss tensor.
    """

    if token_ids.ndim != 2:
        raise ValueError(
            "token_ids must have shape "
            "[batch_size, sequence_length]."
        )

    if logits.shape[:2] != token_ids.shape:
        raise ValueError(
            "The batch and sequence dimensions of logits "
            "must match token_ids."
        )

    prediction_logits = logits[
        :, :-1, :
    ]

    targets = token_ids[
        :, 1:
    ]

    vocab_size = logits.size(-1)


    flat_prediction_logits = (
        prediction_logits.reshape(
            -1,
            vocab_size,
        )
    )

    flat_targets = targets.reshape(
        -1
    )

    loss = F.cross_entropy(
        flat_prediction_logits,
        flat_targets,
    )

    return loss
