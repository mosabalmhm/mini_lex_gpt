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

@torch.no_grad()
def inspect_predictions(
    model: MiniLexGPT,
    token_ids: torch.Tensor,
    label: str,
) -> None:

    model.eval()

    logits = model(
        token_ids
    )

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    prediction_probabilities = probabilities[
        :, :-1, :
    ]

    targets = token_ids[
        :, 1:
    ]

    print(f"\n--- {label} ---")

    for batch_index in range(
        token_ids.size(0)
    ):
        
        print(
            f"\nSequence {batch_index}:"
        )

        for position in range(
            targets.size(1)
        ):

            target_id = targets[
                batch_index,
                position,
            ].item()

            target_probability = (
                prediction_probabilities[
                    batch_index,
                    position,
                    target_id,
                ].item()
            )

            predicted_id = (
                prediction_probabilities[
                    batch_index,
                    position,
                ]
                .argmax()
                .item()
            )

            predicted_probability = (
                prediction_probabilities[
                    batch_index,
                    position,
                    predicted_id,
                ].item()
            )

            context = token_ids[
                batch_index,
                : position + 1,
            ].tolist()

            print(
                f"context={context} | "
                f"target={target_id} | "
                f"target_prob={target_probability:.6f} | "
                f"top_prediction={predicted_id} | "
                f"top_prob={predicted_probability:.6f}"
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

    inspect_predictions(
        model=model,
        token_ids=token_ids,
        label="Before training",
    )

    number_of_steps = 100

    for step in range(
        1,
        number_of_steps + 1,
    ):
        loss, gradient_norm = train_step(
            model=model,
            optimizer=optimizer,
            token_ids=token_ids,
            max_gradient_norm=1.0,
        )

        if step % 20 == 0:
            print(
                f"\nTraining step {step:03d} | "
                f"loss={loss.item():.6f} | "
                f"grad_norm={gradient_norm.item():.6f}"
            )

    inspect_predictions(
        model=model,
        token_ids=token_ids,
        label="After training",
    )

if __name__ == "__main__":
    main()
