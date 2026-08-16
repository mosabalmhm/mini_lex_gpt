from __future__ import annotations

import torch
import torch.nn.functional as F

from mini_lex_gpt.config import ModelConfig
from mini_lex_gpt.model.language_model import MiniLexGPT

def compute_loss(#
    model: MiniLexGPT,
    token_ids: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:

    logits = model(token_ids)
    
    prediction_logits = logits[
        :, :-1, :
    ]

    targets = token_ids[
        :, 1:
    ]

    flat_logits = prediction_logits.reshape(
        -1,
        vocab_size,
    )

    flat_targets = targets.reshape(
        -1
    )

    return F.cross_entropy(
        flat_logits,
        flat_targets,
    )

def main() -> None:
    torch.manual_seed(42)

    config= ModelConfig()

    model = MiniLexGPT(
         config=config,
    )

    model.eval()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
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

    loss_before = compute_loss(
        model=model,
        token_ids=token_ids,
        vocab_size=config.vocab_size,
    )

    optimizer.zero_grad()

    loss_before.backward()

    gradient_before_step = (
        selected_weight.grad[0, 0].item()
    )

    optimizer.step()

    beta1, beta2 = (
        optimizer.param_groups[0]["betas"]
    )
    
    expected_first_moment = (
        (1.0 - beta1)
        * gradient_before_step
    )

    expected_second_moment = (
        (1.0 - beta2)
        * gradient_before_step
        * gradient_before_step
    )

    optimizer_state = optimizer.state[
        selected_weight
    ]

    step_number = optimizer_state[
        "step"
    ]

    first_moment = optimizer_state[
        "exp_avg"
    ]

    second_moment = optimizer_state[
        "exp_avg_sq"
    ]
    weight_after  = (
        selected_weight[0, 0].item()
    )

    loss_after = compute_loss(
        model=model,
        token_ids=token_ids,
        vocab_size=config.vocab_size,
    )

    print("========================")
    print(" MiniLexGPT AdamW step")
    print("========================")

    print("\nLoss before:")
    print(loss_before.item())

    print("\nSelected weight before:")
    print(weight_before)

    print("\nSelected gradient:")
    print(gradient_before_step)

    print("\nSelected weight after:")
    print(weight_after)

    print("\nWEight changed:")
    print(
        weight_after != weight_before
    )
 
    print("\nLoss after:")
    print(loss_after.item())

    print("\nLoss change:")
    print(
        loss_after.item()
        - loss_before.item()
    ) 
  

    print("\nAdamW betas:")
    print(
        beta1,
        beta2,
    )

    print("\nExpected first moment:")
    print(
        expected_first_moment
    )

    print("\nActual first moment:")
    print(
        first_moment[0, 0].item()

    )

    print("\nFirst moment is correct:")
    print(
        abs(
            first_moment[0, 0].item()
            - expected_first_moment
        ) < 1e-10
    )
 
    print("\nExpected second moment:")
    print(
        expected_second_moment
    )

    print("\nActual second moment:")
    print(
        second_moment[0, 0].item()
    )

    print("\nSecond moment is correct:")
    print(
        abs(
            second_moment[0, 0].item()
            - expected_second_moment
        ) < 1e-12
    )

    print("\n--- AdamW Internal State ---")
    
    print("Optimizer step:")
    print(step_number)
    
    print("\nFirst moment shape:")
    print(first_moment.shape)
    
    print("\nSelected first moment:")
    print(
        first_moment[0, 0].item()
    )

    print("\nSecond moment shape:")
    print(second_moment.shape)

    print("\nSelected second moment:")
    print(
        second_moment[0, 0].item()
    )

    print("\nLoss decreased:") 
    print(
        loss_after.item()
        < loss_before.item()
    )


if __name__ == "__main__":
    main()
