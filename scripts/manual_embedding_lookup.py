from __future__ import annotations

import torch 
from torch import nn

class ManualTokenEmbedding(nn.Module):
    """
    
    A small educational implementation of token embedding.

    Instead of using nn.Embedding, this class creates a trainable
    weight matrix and selects rows using token IDs.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(
                "vocab_size mut be greater than zero."
            )

        if d_model <= 0:
            raise ValueError(
                "d_model must be greater than zero."
            )

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.weight = nn.Parameter(
            torch.randn(vocab_size, d_model)
        )



    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select one row from weight matrix for every token IDs.

        Input:
            [batch_size, sequence_length]
        Output:
            [batch_size, sequence_length, d_model]
        """

        if token_ids.dtype != torch.long:
            raise TypeError(
                "token_ids must have dtype torch.long."
            )
        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids must have shape"
                "[batch_size, sequence_length]."
            )

        if token_ids.numel() > 0:
            minimum_id = int(token_ids.min().item())
            maximum_id = int(token_ids.max().item())
 
            if minimum_id < 0:
                raise ValueError(
                    "Token IDs cannot be negative."
                )
            if maximum_id >= self.vocab_size:
                raise ValueError(
                    f"Token ID {maximum_id} is outside "
                    f"vocabulary size {self.vocab_size}."
                )

        # This line performs the embedding liikup.
        return self.weight[token_ids]

def main() -> Nine:
    vocabulary_size = 6
    model_dimension = 3

    layer = ManualTokenEmbedding(
        vocab_size=vocabulary_size,
        d_model=model_dimension,
    )

    fixed_weights = torch.tensor(
        [
            [0.0, 0.1, 0.2], # Token 0
            [1.0, 1.1, 1.2], # Token 1
            [2.0, 2.1, 2.2], # Token 2
            [3.0, 3.1, 3.2], # Token 3
            [4.0, 4.1, 4.2], # Token 4
            [5.0, 5.1, 5.2], # Token 5
        ],
        dtype=torch.float32,
    )


    # Replace the random values with clear fixed values.
    # no_grad prevents this manual copy from entering
    # the automatic-gradient computation graph.
    with torch.no_grad():
        layer.weight.copy_(fixed_weights)
    token_ids = torch.tensor(
        [
            [1, 4, 1],
            [2, 0, 5],
        ],
        dtype=torch.long,
    )

    embedded_tokens = layer(token_ids)
    manually_selected_rows = fixed_weights[token_ids]
    print("==============================")
    print("Manual Token Embedding")
    print("==============================")

    print("\n--- Complete Weight Matrix ---")
    print(layer.weight)

    print("\nWeight shape:")
    print(layer.weight.shape)

    print("\n--- Token IDs ---")
    print(token_ids)
   
    print("\nToken IDs shape:")
    print(token_ids.shape)

    print("\n--- Selected Embedding Vectors ---")
    print(embedded_tokens)

    print("\nEmbedding output shape:")
    print(embedded_tokens.shape)
 
    print("\n--- MAnual Lookup Check ---")
    
    outputs_are_equal = torch.allclose(
        embedded_tokens,
        manually_selected_rows,
    )

    print(
        "Layer output equals direct row selection:",
        outputs_are_equal,
    )

    print("\nToken ID 1 selects this row:")
    print(layer.weight[1])

    print("\nFirst Token ID in the batch:")
    print(token_ids[0, 0])

    print("\nIts resulting vector:")
    print(embedded_tokens[0, 0])

    print("\n--- Gradient Experiment ---")

    loss = embedded_tokens.sum()

    print("Artificial loss:")
    print(loss.item())

    loss.backward()

    print("\nGradient matrix:")
    print(layer.weight.grad)

    print("\nGradient for Token 1:")
    print(layer.weight.grad[1])

    print("\nGradient for unused Token 3:")
    print(layer.weight.grad[3])

    print("\n=======================================")
    print(" Manual embedding experiment completed")
    print("=========================================")

if __name__ == "__main__":
    main()
