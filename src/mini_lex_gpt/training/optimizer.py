from __future__ import annotations

import torch
from torch import nn


def build_adamw_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> torch.optim.AdamW:

    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:
            continue

        if parameter.dim() >= 2:
            decay_parameters.append(
                parameter
            )

        else:
            no_decay_parameters.append(
                parameter
            )

    trainable_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }

    decay_ids = {
        id(parameter)
        for parameter in decay_parameters
    }

    no_decay_ids = {
        id(parameter)
        for parameter in no_decay_parameters
    }

    overlapping_ids = (
        decay_ids
        & no_decay_ids
    )

    grouped_ids = (
        decay_ids
        | no_decay_ids
    )

    missing_ids = (
        trainable_ids
        - grouped_ids
    )

    if len(overlapping_ids) != 0:
        raise RuntimeError(
            "A trainable parameter was assigned "
            "to more than one optimizer group."
        )

    if len(missing_ids) != 0:
        raise RuntimeError(
            "Some trainable parameters were not "
            "assigned to an optimizer group."
        )
    parameter_groups = [
        {
            "params": decay_parameters,
            "weight_decay": weight_decay,
        },
        {
            "params": decay_parameters,
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
    )


    return optimizer

