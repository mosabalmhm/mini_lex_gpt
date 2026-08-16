from __future__ import annotations

import torch
import torch.nn.functional as F


from mini_lex_gpt.config import ModelConfig
from mini_lex_gpt.model.language_model import MiniLexGPT

def compute_loss(
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
    
    model.eval()

    token_ids = torch.tensor(
        [
            [1, 25, 90, 500, 3],
            [1, 70, 90, 800, 3],
        ],
        dtype=torch.long,
    )

    learning_rate = 0.001

    qkv_weight = (
        model.blocks[0]
        .attention
        .qkv
        .qkv_projection
        .weight
    )

    loss_before = compute_loss(
        model=model,
        token_ids=token_ids,
        vocab_size=config.vocab_size,
    )

    weight_before = (
        qkv_weight[0, 0].item()
    )

    loss_before.backward()

    gradient_value = (
        qkv_weight.grad[0, 0].item()
    )

    gradient_norm = (
        qkv_weight.grad.norm().item()
    )

    expected_new_weight = (
        weight_before
        - learning_rate * gradient_value
    )

    with torch.no_grad():
        qkv_weight -= (
            learning_rate
            * qkv_weight.grad
        )

    weight_after = (
        qkv_weight[0, 0].item()
    )
    
    loss_after = compute_loss(
        model=model,
        token_ids=token_ids,
        vocab_size=config.vocab_size,
    )

    print("===================================")
    print(" MiniLexGPT Manual Weight Update")
    print("===================================")

    print("\n--- Learning Rate ---")
    
    print("Learning rate:")
    print(learning_rate)

    print("\n--- Before Update ---")
    print("Loss before update:")
    print(loss_before.item())

    print("\nSelected QKV weight before:")
    print(weight_before)

    print("\nSelected weight gradient:")
    print(gradient_value)

    print("\nFull QKV graadient norm:")
    print(gradient_norm)

    print("\n--- Manual Calculation ---")

    print("Expected selected weight after update:")
    print(expected_new_weight)

    print("Actual selected weight after update:")
    print(weight_after)

    print(
        "\nManual calculation matches actual update:"
    )

    print(
        abs(
            expected_new_weight
            - weight_after
        ) < 1e-7
    )

    print("\nLoss after update:")
    print(loss_after.item())

    print("\nLoss change:")
    print(
        loss_after.item()
        - loss_before.item()
    )

    print("\nLoss decreased:")
    print(
        loss_after.item()
        < loss_before.item()
    )


    print("\n==================================")
    print(" Manual update inspection completed")
    print("=====================================")

if __name__ == "__main__":
    main()
