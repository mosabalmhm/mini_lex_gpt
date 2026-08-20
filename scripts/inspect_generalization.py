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

def build_datasets() -> tuple[
    torch.Tensor,
    torch.Tensor,
]:

    a_tokens = [10, 11, 12, 13]
    b_tokens = [20, 21, 22, 23]

    held_out_pairs = {
        (10, 20),
        (11, 21),
        (12, 22),
        (13, 23),
    }

    training_sequences = []
    validation_sequences = []

    for a_token in a_tokens:
        for b_token in b_tokens:

            sequence = [
                1,
                a_token,
                b_token,
                99,
                a_token,
                b_token,
                3,
            ]

            if (
                a_token,
                b_token,
            ) in held_out_pairs:

                validation_sequences.append(
                    sequence
                )

            else:
                training_sequences.append(
                    sequence
                )

    return (
        torch.tensor(
            training_sequences,
            dtype=torch.long,
        ),
        torch.tensor(
            validation_sequences,
            dtype=torch.long,
        ),
    )

@torch.no_grad()
def evaluate_copy_accuracy(
    model: MiniLexGPT,
    token_ids: torch.Tensor,
) -> tuple[float, float]:

    model.eval()

    logits = model(
        token_ids
    )

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    #Position 3 predicts the copied A
    first_copy_logits = logits[
        :, 3, :
    ]

    # Position 4 predicts the copied B
    second_copy_logits = logits[
        :, 4, :
    ]

    first_targets = token_ids[
        :, 4
    ]

    second_targets = token_ids[
        :, 5
    ]

    first_predictions = (
        first_copy_logits.argmax(
            dim=-1
        )
    )

    second_predictions = (
        second_copy_logits.argmax(
            dim=-1
        )
    )

    correct = (
        (first_predictions == first_targets)
        .float()
        .sum()
        +
        (second_predictions == second_targets)
        .float()
        .sum()
    )

    total = (
        token_ids.size(0) * 2
    )

    accuracy = (
        correct.item() / total
    )

    row_indices = torch.arange(
        token_ids.size(0)
    )

    first_target_probabilities = (
        probabilities[
            row_indices,
            3,
            first_targets,
        ]
    )

    second_target_probabilities = (
        probabilities[
            row_indices,
            4,
            second_targets,
        ]
    )

    mean_target_probability = (
        torch.cat(
            [
                first_target_probabilities,
                second_target_probabilities,
            ]
        )
        .mean()
        .item()
    )

    return (
        accuracy,
        mean_target_probability,
    )

@torch.no_grad()
def print_validation_predictions(
    model: MiniLexGPT,
    validation_tokens: torch.Tensor,
) -> None:

    model.eval()

    logits = model(
        validation_tokens
    )

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    print("\n--- Held-out combinations ---")

    for row in range(
        validation_tokens.size(0)
    ):

        sequence = (
            validation_tokens[row]
            .tolist()
        )

        target_a = sequence[4]
        target_b = sequence[5]

        predicted_a = (
            logits[row, 3]
            .argmax()
            .item()
        )

        predicted_b = (
            logits[row, 4]
            .argmax()
            .item()
        )

        probability_a = (
            probabilities[
                row,
                3,
                target_a,
            ]
            .item()
        )

        probability_b = (
            probabilities[
                row,
                4,
                target_b,
            ]
            .item()
        )

        print(
            f"input={sequence[:4]} | "
            f"expected=[{target_a}, {target_b}] | "
            f"predicted=[{predicted_a}, {predicted_b}] | "
            f"pA={probability_a:.4f} | "
            f"pB={probability_b:.4f}"
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

    (
        training_tokens,
        validation_tokens,
    ) = build_datasets()

    print("=============================")
    print(" MiniLexGPT Generalization")
    print("=============================")

    print("\nTraining sequences:")
    print(training_tokens.size(0))

    print("\nHeld-out sequences:")
    print(validation_tokens.size(0))

    (
        initial_accuracy,
        initial_probability,
    ) = evaluate_copy_accuracy(
       model=model,
       token_ids=validation_tokens,
    )

    print("\nBefore training:")
    print(
        f"Held-out copy accuracy:"
        f"{initial_accuracy:.2%}"
    )
    print(
        f"Mean correct probability: "
        f"{initial_probability:.6f}"
    )

    number_of_steps = 300

    for step in range(
        1,
        number_of_steps + 1,
    ):

        loss, gradient_norm = train_step(
            model=model,
            optimizer=optimizer,
            token_ids=training_tokens,
            max_gradient_norm=1.0,
        )

        if step % 50 == 0:
            (
                validation_accuracy,
                validation_probability,
            ) = evaluate_copy_accuracy(
                model=model,
                token_ids=validation_tokens,
            )

            print(
                f"\nSteo {step:03d} | "
                f"loss={loss.item():.6f} | "
                f"grad_norm={gradient_norm.item():.6f} | "
                f"held_out_accuracy="
                f"{validation_accuracy:.2%} | "
                f"held_out_prob="
                f"{validation_probability:.4f}"
            )

    print_validation_predictions(
        model=model,
        validation_tokens=validation_tokens,
    )

if __name__ == "__main__":
    main()
