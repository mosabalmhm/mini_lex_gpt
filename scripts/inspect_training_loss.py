from __future__ import annotations

import torch

from mini_lex_gpt.config import ModelConfig
from mini_lex_gpt.model.language_model import MiniLexGPT
from mini_lex_gpt.training.loss import (
    next_token_cross_entropy,
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

    loss = next_token_cross_entropy(
        logits=logits,
        token_ids=token_ids,
    )

    prediction_logits = logits[
        :, :-1, :
    ]

    targets = token_ids[
        :, 1:
    ]

    print("==============================")
    print(" MiniLexGPT Training Loss")
    print("===============================")

    print("\nToken IDs shape:")
    print(token_ids.shape)

    print("\nLogits shape:")
    print(logits.shape)
  
    print("\nPrediction logits shape:")
    print(prediction_logits.shape)

    print("\nTargets shape:")
    print(targets.shape)

    print("\nVocabulary size:")
    print(logits.size(-1))

    print("\nNumber of prediction positions:")
    print(targets.numel())
 
    print("\nLoss shape:")
    print(loss.shape)

    print("\nLoss")
    print(loss.item())

    print("\nLoss requires gradient:")
    print(loss.requires_grad)
    
    print("\nLoss is finite:")
    print(
        torch.isfinite(loss).item()
    ) 

    print("\n=============================")
    print(" Loss inspection completed")
    print("===============================")


if __name__ == "__main__":
    main()
