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



def compute_global_gradient_norm(
    model: MiniLexGPT,
) -> torch.Tensor:

    gradient_norms = []
    
    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        parameter_gradient_norm = (
            parameter
            .grad
            .detach()
            .float()
            .norm(2)
        )

        gradient_norms.append(
            parameter_gradient_norm
        )

    if len(gradient_norms) == 0:
        return torch.tensor(0.0)

    stacked_norms = torch.stack(
        gradient_norms
    )

    global_norm = stacked_norms.norm(
        2
    )


    return global_norm

def main() -> None:
    torch.manual_seed(42)

    config = ModelConfig()

    model = MiniLexGPT(
        config=config,
    )

    model.train()

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

    max_gradient_norm = 1.0

    optimizer.zero_grad(
        set_to_none=True
    )

    loss = compute_loss(
        model=model,
        token_ids=token_ids,
        vocab_size=config.vocab_size,
    )

    loss.backward()

    qkv_weight = (
        model.blocks[0]
        .attention
        .qkv
        .qkv_projection
        .weight
    )

    global_norm_before = (
        compute_global_gradient_norm(
            model
        )
    )
 
    qkv_norm_before = (
        qkv_weight.grad.norm().item()
    )

    returned_norm = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_gradient_norm,
            norm_type=2.0,
            error_if_nonfinite=True,
        )
    )

    global_norm_after = (
        compute_global_gradient_norm(
            model
        )
    )

    qkv_norm_after = (
        qkv_weight.grad.norm().item()
    )


    print("================================")
    print(" MiniLexGPT Gradient Clipping")
    print("================================")

    print("\nLoss:")
    print(
        loss.item()
    )

    print("\nMaximum gradient norm:")
    print(
        max_gradient_norm
    )

    print("\n--- Before Clipping ---")
    
    print("Global gradient norm:")
    print(
        global_norm_before.item()
    )

    print("\nQKV gradient norm:")
    print(
        qkv_norm_before
    )

    print("\n--- PyTorch clip_grad_norm_ ---")

    print("Returned total norm:")
    print(returned_norm.item())

    print("\nReturned norm matches manual norm:")
    print(
        abs(
            returned_norm.item()
            - global_norm_before.item()
        ) < 1e-5
    )

    print("\n--- After Clipping ---")

    print("Global gradient norm :")
    print(
        global_norm_after.item()
    )

    print("\nQKV gradient norm:")
    print(
        qkv_norm_after
    )

    print("\nGlobal norm is within maximum:")
    print(
        global_norm_after.item()
        <= max_gradient_norm + 1e-5
    )

    print("\nQKV gradient was scaled:")
    print(
        qkv_norm_after
        < qkv_norm_before
    )

    print("\nAll gradient finite:")
    print(
        torch.isfinite(
            global_norm_after
        ).item()
    )

    print("\n==================================")
    print(" Gradient clipping inspection completed")
    print("=========================================")


if __name__ == "__main__":
    main()
