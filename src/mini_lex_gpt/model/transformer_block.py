from __future__ import annotations

import torch
from torch import nn

from mini_lex_gpt.config import ModelConfig
from mini_lex_gpt.model.attention import (
    CausalGroupedQueryAttention,
)
from mini_lex_gpt.model.feed_forward import (
    SwiGLUFeedForward,
)
from mini_lex_gpt.model.rmsnorm import RMSNorm

class TransformerBlock(nn.Module):
    """
    
    Pre-RMSNorm decoder Teansformer block.

    Structure:
        x
        |-- RMSNorm -> Attenstion __
        ---------------------------+ -> X1
    
        X1
        |---RMSNorm -> SwiGLU ----
        -------------------------+ -> Output

    Input shape:
        [batch_size, sequence_length, d_model]

    Output shape:
        [batch_size, sequence_length, d_model]
    """

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()

        self.d_model = config.d_model

        self.attention_norm =RMSNorm(
            dimension=config.d_model,
            epsilon=config.rms_norm_epsilon,
        )

        self.attention = CausalGroupedQueryAttention(
            config=config,
        )

        self.feed_forward_norm = RMSNorm(
            dimension=config.d_model,
            epsilon=config.rms_norm_epsilon,
        )

        self.feed_forward = SwiGLUFeedForward(
            config=config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """

        Run one Transformer block.
        """

        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape "
                "[batch, sequence, d_model]."
            )

        if hidden_states.shape[-1] != self.d_model:
            raise ValueError(
                "The final hidden-state dimension must equal "
                f"d_model={self.d_model}, but received "
                f"{hidden_states.shape[-1]}."
            )

        if not hidden_states.is_floating_point():
            raise TypeError(
                "hidden_states must use a floating-point dtype"
            )

        #------------------------------
        # Attention sublayer
        #------------------------------
        
        residual = hidden_states
        
        normalized_states = self.attention_norm(
            hidden_states
        )

        attention_output = self.attention(
            normalized_states,
            position_offset=position_offset,
        )

        hidden_states = (
            residual + attention_output
        )

        #-------------------------
        # Feed-forward sublayer
        #-------------------------

        residual = hidden_states

        normalized_states = self.feed_forward_norm(
            hidden_states
        )

        feed_forward_output = self.feed_forward(
            normalized_states
        )

        hidden_states = (
            residual + feed_forward_output
        )

        return hidden_states
