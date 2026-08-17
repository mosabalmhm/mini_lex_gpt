from __future__ import annotations

import torch

from mini_lex_gpt.config import ModelConfig
from mini_lex_gpt.model.language_model import MiniLexGPT
from mini_lex_gpt.training.loss import (
    next_token_cross_entropy,
)
from mini_lex_gpt.training.optimizer import (
    build_adamw_optimizer,
)
from mini_lex_gpt.training.train_step import (
    train_step,
)

@torch.no_grad()
def evaluate_loss(
    model: MiniLexGPT,
    token_ids: torch.Tensor,
) -> torch.Tensor:

    model.eval()

    logits = model(
        token_ids
    )

    loss = next_token_cross_entropy(
        logits = logits,
        token_ids=token_ids,
    )
          
    return loss

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

    number_of_steps = 20
    
    initial_eval_loss = evaluate_loss(
      model=model,
      token_ids=token_ids,
    )

    print("=================================")
    print(" MiniLexGPT Multi-Step training")
    print("=================================")

    print("\nInitial evaluation loss:")
    print(
        initial_eval_loss.item()
    )

    print("\n--- Training ---")
    for step in range(
        1,
        number_of_steps + 1,
    ):
 
        training_loss, gradient_norm = train_step(
            model=model,
            optimizer=optimizer,
            token_ids=token_ids,
            max_gradient_norm=1.0,
        )

        evaluation_loss = evaluate_loss(
            model=model,
            token_ids=token_ids,
        )

        print(
            f"Step {step:02d} | "
            f"train_loss={training_loss.item():.6f} | "
            f"eval_loss={evaluation_loss.item():.6f} | "
            f"grad_norm={gradient_norm.item():.6f}"
        )


    final_eval_loss = evaluate_loss(
        model=model,
        token_ids=token_ids,
    )

    print("--- Result ---")

    print("\nInitial evaluation loss:")
    print(
        initial_eval_loss.item()
    )

    print("\nFinal evaluation loss:")
    print(
        final_eval_loss.item()
    )
    
    print("\nLoss decrease:")
    print(
        initial_eval_loss.item()
        - final_eval_loss.item()
    )

    print("\nModel learned the batch:")
    print(
        final_eval_loss.item()
        < initial_eval_loss.item()
    )

    print("===============================")
    print(" Multi-step learning completed")
    print("================================")

if __name__ == "__main__":
    main()
