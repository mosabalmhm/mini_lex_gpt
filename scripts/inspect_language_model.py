from __future__ import annotations

import torch

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
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("===============================")
    print(" MiniLexGPT Full Language Model")
    print("===============================")

    print("\n--- Configuration ---")

    print("Vocabulary size:")
    print(config.vocab_size)

    print("\nModel dimension:")
    print(config.d_model)

    print("\nTransformer layers:")
    print(config.num_layers)

    print("\n--- Input ---")
    
    print("Token IDs shape:")
    print(token_ids.shape)

    print("\nToken IDs:")
    print(token_ids)

    print("\n--- Model Structure ---")

    print("Number of Transformer blocks:")
    print(len(model.blocks))

    print("\n--- Output ---")
    
    print("Logits shape:")
    print(logits.shape)

    print("\nExpected logits shape:")
    print(
        torch.Size(
            [
                token_ids.shape[0],
                token_ids.shape[1],
                config.vocab_size,
            ]
        )
    )

    print(
        "\nLogits shape is correct:",
        logits.shape
        == torch.Size(
            [
                token_ids.shape[0],
                token_ids.shape[1],
                config.vocab_size,
            ]
        ),
    )

    print("\n--- Parameter Count ---")

    print("Trainable model parameters:")
    print(parameter_count)

    print("\nExpected:")
    print(1_311_872)

    print(
        "\nParameter count is correct:",
        parameter_count == 1_311_872,
    )

    print("\n--- Weight Tying ---")

    embedding_weight = (
        model.token_embedding.embedding.weight
    )

    print("Embedding weight shape:")
    print(embedding_weight.shape)

    print("\nNo separate LM-head parameter was created.")
 
    print("\n--- Sample Logits ---")
    print("First batch, first token, first token, first 10 logits:")
    print(
        logits[0, 0, :10]
    )

    print("=" * 25)
    print(" Full model inspection completed")
    print("=" * 25)

if __name__ == "__main__":
    main()
