from __future__ import annotations

import torch
import torch.nn.functional as F

from mini_lex_gpt.config import ModelConfig
from mini_lex_gpt.model.language_model import (
    MiniLexGPT,
)

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
            config.vocab_size,
        )
    )

    flat_targets = targets.reshape(
        -1
    )

    loss = F.cross_entropy(
        flat_prediction_logits,
        flat_targets,
    )

    print("==============================")
    print(" MiniLexGPT Next-Token Loss")
    print("==============================")

    print("\n--- Original Tokens ---")
    
    print("Token IDs:")
    print(token_ids)

    print("\nTkoen IDS shape:")
    print(token_ids.shape)

    print("\n--- Model Output ---")

    print("Full logits shape:")
    print(logits.shape)

    print("\n--- Shifted Training Date ---")

    print("Prediction logits shape:")
    print(prediction_logits.shape)

    print("\nTargets:")
    print(targets)
    
    print("\nTarget shape:")
    print(targets.shape)

    print("\n--- Flattened Shapes ---")

    print("Flattened prediction logits shape:")
    print(flat_prediction_logits.shape)
    print("\nFlattened Targets:")
    print(flat_targets)

    print("\nFlattened targets shape:")
    print(flat_targets.shape)

    print("\nNumber of prediction tasks:")
    print(flat_targets.numel())

    print("\n--- Cross-Entropy Loss ---")

    print("Loss:")
    print(loss)

    print("\nLoss shape:")
    print(loss.shape)

    print("\nLoss is finite:")
    print(
        torch.isfinite(loss).item()
    )

    print("\n====================================")
    print(" Next-token loss inspection completed")
    print("======================================")

if __name__ == "__main__":
    main()
