from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mini_lex_gpt.config import ModelConfig
from mini_lex_gpt.model.embedding import TokenEmbedding
from mini_lex_gpt.model.rmsnorm import RMSNorm
from mini_lex_gpt.model.transformer_block import (
    TransformerBlock,
)


class MiniLexGPT(nn.Module):
    """
 
    Decoder-only Transformer language model.

    Input:
        token_ids[B, T]

    Output:
        logits [B, T, vocab_size]
    """

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.token_embedding = TokenEmbedding(
            config=config,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config=config)
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = RMSNorm(
            dimension=config.d_model,
            epsilon=config.rms_norm_epsilon,
        )
        
        self.apply(
            self._initialize_weights
        )
    
    def _initialize_weights(
        self,
        module: nn.Module,
    ) -> None:
        
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_std,
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_std,
            )

    def forward(
        self,
        token_ids: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """

        Convert token IDs into vocabulary logits.
        """

        hidden_states = self.token_embedding(
            token_ids
        )

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                position_offset=position_offset,
            )

        hidden_states = self.final_norm(
            hidden_states
        )

        logits = F.linear(
            hidden_states,
            self.token_embedding.embedding.weight,
        )

        return logits
