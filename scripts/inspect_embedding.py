from __future__ import annotations

import torch 
from torch import nn

from mini_lex_gpt.config import ModelConfig
from mini_lex_gpt.model.embedding import TokenEmbedding

def count_trainable_parameters(module: nn.Module) -> int:
    """
    
    Count parameters that will be updated during training.
    """

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )

def main() -> None:
    torch.manual_seed(42)
   
    config = ModelConfig()
    token_embedding = TokenEmbedding(config)

    token_ids = torch.tensor(
        [
            [1, 25, 90, 500, 3],
            [1, 70, 90, 800, 3],
        ],
        dtype=torch.long,
    )

    embedded_tokens = token_embedding(token_ids)
    parameter_count = count_trainable_parameters(
        token_embedding
    )

    parameter_memory_mib = (
        token_embedding.embedding.weight.numel()
        * token_embedding.embedding.weight.element_size()
        / (1024 ** 2)
    )



    print("=" * 20)
    print("MiniLexGPT Token Embedding")
    print("=" * 20)

    print("\n--- Layer ---")
    print(token_embedding)

    print("\n---- Token IDs ----")
    print(token_ids)


    print("\nToken ID shape:")
    print(token_ids.shape)

    print("\n--- Embedding Output ---")
    print("Embedding output shape:")
    print(embedded_tokens.shape)

    print("\nEmbedding data type:")
    print(embedded_tokens.dtype)

    print("\nEmbedding device:")
    print(embedded_tokens.device)

    print("\nFirst token, first 8 dimensions:")
    print(
        embedded_tokens[0, 0, :8].detach()
    )

    print("\n--- Equality Checks ---")

    same_token_same_vector = torch.allclose(
        embedded_tokens[0, 0],
        embedded_tokens[1, 0],
    )

    repeated_token_same_vector = torch.allclose(
        embedded_tokens[0, 2],
        embedded_tokens[1, 2],
    )
  
    different_tokens_different_vectors = not torch.allclose(
        embedded_tokens[0, 1],
        embedded_tokens[1, 1],
    )

    print(
        "Token ID 90 produces the same vector:",
        repeated_token_same_vector,
    )

    print(
        "Different IDs produce different vectors:",
        different_tokens_different_vectors,
    )

    print("\n--- Parameters ---")
    print(
        "Embedding weight shape:",
        tuple(token_embedding.embedding.weight.shape),
    )


    print(
        "Trainable parameters:",
        f"{parameter_count:,}",
    )

    print(
        "Parameter memory, float32:",
        f"{parameter_memory_mib:.2f} MiB",
    )
  
    print(
        "Weight require gradients:",
        token_embedding.embedding.weight.requires_grad,
    )


    print("\n================================")
    print(" Embedding inspection completed")
    print("==================================")

if __name__ == "__main__":
    main()
