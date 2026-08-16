from __future__ import annotations

import torch
from torch import nn

from mini_lex_gpt.config import ModelConfig

class TokenEmbedding(nn.Module):
    """
    convert integer token IDs into trainable dense vectors.
    Input shape:
        [batch size, sequence_length]
    output_shape:
        [batch_size, squence_length, d_model]
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
  
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up the embedding vector for every token Id.
        """
    
        if token_ids.dtype != torch.long:
            raise ValueError(
                "token_ids must use torch.long (int64) "
                f"but received {token_ids.dtype}."
            )

        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids must have shape "
                "[batch_size, sequence_length], "
                f"but received shape {tuple(token_ids.shape)}."
            )
  
        if token_ids.numel() > 0:
            smallest_token_id = int(token_ids.min().item())
            largest_token_id = int(token_ids.max().item())

            if smallest_token_id < 0:
                raise ValueError(
                    "Token IDs cannot be negative. "
                    f"Smallest received ID: {smallest_token_id}."
                )

            if largest_token_id >= self.vocab_size:
                raise ValueError(
                    "A token ID is outside the vocabulary. "
                    f"Largest received ID: {largest_token_id}; "
                    f"vocabulary size: {self.vocab_size}."
                )
        return self.embedding(token_ids)  
