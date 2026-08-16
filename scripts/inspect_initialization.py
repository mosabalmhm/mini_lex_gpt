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

    embedding_weight = (
        model.token_embedding.embedding.weight
    )

    print("=================================")
    print(" MiniLexGPT Weight Initialization")
    print("=================================")

    print("\n--- Embedding Weight ---")
    
    print("Shape:")
    print(embedding_weight.shape)

    print("\nmean:")
    print(
        embedding_weight.mean().item()
    )

    print("\nStandard deviation:")
    print(
        embedding_weight.std().item()
    )


    print("\nMinimum value:")
    print(
        embedding_weight.min().item()
    )
    
    print("\nMaximum value:")
    print(
        embedding_weight.max().item()
    )

    first_linear_weight = (
        model.blocks[0]
        .attention
        .qkv
        .qkv_projection
        .weight
    )

    print("\n--- First QKV Linear Weight ---")
    
    print("shape:")
    print(first_linear_weight.shape)

    print("\nMean:")
    print(
        first_linear_weight.mean().item()
    )

    print("\nStandard deviation:")
    print(
        first_linear_weight.std().item()
    )

    print("\nAll embedding values finite:")
    print(
        torch.isfinite(
            embedding_weight
        ).all().item()
    )


    print("\n==============================")
    print(" Initialization inspection completed")
    print("======================================")


if __name__ == "__main__":
    main()
