from __future__ import annotations

import torch
import torch.nn.functional as F

from mini_lex_gpt.config import ModelConfig
from mini_lex_gpt.model.language_model import MiniLexGPT
from mini_lex_gpt.training.optimizer import (
    build_adamw_optimizer,
)

def compute_next_token_loss(
    model: MiniLexGPT,
    token_ids: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:

    logits = model(
        token_ids
    )

    prediction_logits = logits[
        :, :-1, :
    ]

    targets = token_ids[
        :, 1:
    ]

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

def main() -> None:
    torch.manual_seed(42)

    config = ModelConfig()
    model = MiniLexGPT(
        config=config,
    )

    model.train()

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

    qkv_weight = (
        model.blocks[0]
        .attention
        .qkv
        .qkv_projection
        .weight
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    gradient_before_backward = (
        qkv_weight.grad
    )

    weight_before = (
        qkv_weight[0, 0].item()
    )

    loss = compute_next_token_loss(
        model=model,
        token_ids=token_ids,
        vocab_size=config.vocab_size,
    )

    loss.backward()

    qkv_gradient_norm_before_clipping = (
        qkv_weight.grad.norm().item()
    )

    total_gradient_norm_before_clipping = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            norm_type=2.0,
            error_if_nonfinite=True,
        )
    )

    qkv_gradient_norm_after_clipping= (
        qkv_weight.grad.norm().item()
    )

    optimizer.step()

    weight_after = (
        qkv_weight[0, 0].item()
    )

    gradient_exists_after_step = (
        qkv_weight.grad is not None
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    gradient_is_none_after_zero_grad = (
        qkv_weight.grad is None
    )

    print("==================================")
    print(" MiniLexGPT Complete Training Step")
    print("==================================")

    print("\nToken IDs shape:")
    print(token_ids.shape)

    print("\nLoss:")
    print(loss.item())

    print("\n--- Gradient lifecycle ---")

    print("\nGradient before backward:")
    print(gradient_before_backward)

    print("\nQKV gradient norm before clipping:")
    print(
        qkv_gradient_norm_before_clipping
    )

    print("\nGlobal gradient norm before clipping:")
    print(
        total_gradient_norm_before_clipping.item()
    )

    print("\nQKV gradient norm after clipping:")
    print(  
        qkv_gradient_norm_after_clipping
    )

    print("\n--- Parameter update ---")

    print("\nSelected weight before:")
    print(weight_before)

    print("\nWeight changed:")
    print(
        weight_before != weight_after
    )

    print("\n--- zero_grad behavior ---")

    print("\nGradient still exists optimizer.step():")
    print(
        gradient_exists_after_step
    )

    print(
        "\nGradient is None after "
        "zero_grad(set_to_none=True):"
    )

    print(
        gradient_is_none_after_zero_grad
    )

    print("\n====================================")
    print(" Training step inspection completed")
    print("======================================")


if __name__ == "__main__":
    main()
