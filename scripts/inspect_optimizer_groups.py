from __future__ import annotations

import torch

from mini_lex_gpt.config import ModelConfig
from mini_lex_gpt.model.language_model import MiniLexGPT
from mini_lex_gpt.training.optimizer import (
    build_adamw_optimizer,
)


def main() -> None:
    torch.manual_seed(42)

    config = ModelConfig()

    model = MiniLexGPT(
        config=config,
    )

    optimizer = build_adamw_optimizer(
        model=model,
        learning_rate=1e-3,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
    )


    print("================================")
    print(" MiniLexGPT optimizer Groups")
    print("================================")

    print("\nNumber of optimizer groups:")
    print(
        len(
            optimizer.param_groups
        )
    )


    print("\n--- Parameter Groups ---")

    for group_index, group in enumerate(
        optimizer.param_groups
    ):

        parameters = group[
            "params"
        ]

        parameter_count = sum(
            parameter.numel()
            for parameter in parameters
        )

        print(
            f"\nGroup {group_index}:"
        )

        print("Weight decay:")
        print(
            group["weight_decay"]
        )

        print("Number of tensors:")
        print(
            len(parameters)
        )

        print("Number of parameters:")
        print(
            parameter_count
        )

    print("\n--- Model Parameters ---")

    total_model_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_optimizer_parameters = sum(
        parameter.numel()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


    print("Trainable model parameters:")
    print(
        total_model_parameters
    )

    print("\nOptimizer parameters:")
    print(
        total_optimizer_parameters
    )

    print("All trainable parameters included:")
    print(
        total_model_parameters
        == total_optimizer_parameters
    )

    print("\n--- Individual Parameters ---")
    
    parameter_group_lookup = {}

    for group_index, group in enumerate(
        optimizer.param_groups
    ):
        for parameter in group[
            "params"
        ]:
            parameter_group_lookup[
                id(parameter)
            ] = group_index

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:
            continue

        group_index = (
            parameter_group_lookup[
                id(parameter)
            ]
        )
        
        print(
            name,
            "shape=",
            tuple(parameter.shape),
            "dim=",
            parameter.dim(),
            "group=",
            group_index,
        )

    print("\n======================================")
    print(" Optimizer group inspection completed")
    print("=======================================")

if __name__ == "__main__":
    main()
