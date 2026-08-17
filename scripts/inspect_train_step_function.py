from __future__ import annotations

import torch

from mini_lex_gpt.config import ModelConfig
from mini_lex_gpt.model.language_model import MiniLexGPT
from mini_lex_gpt.training.optimizer import (
    build_adamw_optimizer,
)
from mini_lex_gpt.training.train_step import (
    train_step,
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


    token_ids = torch.tensor(
        [
            [1, 25, 90, 500, 3],
            [1, 70, 90, 800, 3],
        ],
        dtype=torch.long,
    )

    selected_weight = (
        model.blocks[0]
        .attention
        .qkv
        .qkv_projection
        .weight
    )


    weight_before = (
        selected_weight[0, 0].item()
    )

    loss, gradient_norm = train_step(
        model=model,
        optimizer=optimizer,
        token_ids=token_ids,
        max_gradient_norm=1.0,
    )

    weight_after = (
        selected_weight[0, 0].item()
    )

    print("=============================")
    print(" MiniLexGPT train_step()")
    print("=============================")

    print("\nLoss:")
    print(loss.item())

    print("\nLoss requires gradient:")
    print(loss.requires_grad)

    print("\nGradient norm before clipping:")
    print(
        gradient_norm.item()
    )

    print("\nSelected weight before:")
    print(weight_before)

    print("\nSelected weight after:")
    print(weight_after)

    print("\nWeight changed:")
    print(
        weight_before != weight_after
    )

    print("\nModel is in training mode:")
    print(
        model.training
    )

    print("\n Gradient still exists after step:")
    print(
        selected_weight.grad is not None
    )
    

    print("\nLoss is finite:")
    print(
        torch.isfinite(loss).item()
    )

    print("\nGradient norm is finite:")
    print(
        torch.isfinite(
            gradient_norm
        ).item()
    )


    print("\n================================")
    print(" train_step inspection completed")
    print("==================================")

if __name__ == "__main__":
    main()
