from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mini_lex_gpt.config import ModelConfig
from mini_lex_gpt.model.qkv_projection import (
    GroupedQueryProjection,
)
from mini_lex_gpt.model.rope import (
    RotaryPositionEmbedding,
)

class CausalGroupedQueryAttention(nn.Module):
    """

    Causal self-attention using Grouped-Query Attention.

    Input shape:
        [batch_sie, sequence_length, d_model]

    Output shape:
        [batch_size, sequence_length, d_model]
    """

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()

        self.d_model = config.d_model
        self.num_query_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dimension = config.head_dimension
        self.dropout = config.dropout

        self.query_heads_per_kv_head = (
            config.query_heads_per_kv_head
        )

        self.qkv = GroupedQueryProjection(
            config=config,
        )

        self.rope = RotaryPositionEmbedding(
            head_dimension=config.head_dimension,
            max_sequence_length=config.context_length,
            base=config.rope_base,
        )

        self.output_projection = nn.Linear(
            in_features=config.d_model,
            out_features=config.d_model,
            bias=config.use_bias
        )

    def _expand_kv_heads(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """

        Expand Key or Value heads to match Query heads.
 
        Before:
            [B, Hkv, T, Dh]
        After:
            [B, Hq, T, Dh]
        """

        return tensor.repeat_interleave(
            self.query_heads_per_kv_head,
            dim=1,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """

        Run causal Grouped-Query Self-Attention.
        """

        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape"
                "[batch, sequence, d_model], "
                f"but received {tuple(hidden_states.shape)}."
            )

        if hidden_states.shape[-1] != self.d_model:
            raise ValueError(
                "The final hidden-states dimension must equal "
                f"d_model={self.d_model}, "
                f"but received {hidden_states.shape[-1]}."
            )

        if not hidden_states.is_floating_point():
            raise TypeError(
                "hidden_states must use a floating-point dtype."
            )

        batch_size = hidden_states.shape[0]
        sequence_length = hidden_states.shape[1]

        query, key, value = self.qkv(
            hidden_states
        )
 
        query, key = self.rope(
            query=query,
            key=key,
            position_offset=position_offset,
        )

        expanded_key = self._expand_kv_heads(
            key
        )

        expanded_value = self._expand_kv_heads(
            value
        )

        dropout_probability = (
            self.dropout
            if self.training
            else 0.0
        )

        attention_output = (
            F.scaled_dot_product_attention(
                query=query,
                key=expanded_key,
                value=expanded_value,
                attn_mask=None,
                dropout_p=dropout_probability,
                is_causal=True,
            )
        )

        # [B, Hq, T, Dh]
        # to
        # [B, T, Hq, Dh]
        attention_output = attention_output.transpose(
            1,
            2,
        )
        
        # [B, T, Hq, Dh]
        # to
        # [B, T, d_model]
        attention_output = attention_output.reshape(
            batch_size,
            sequence_length,
            self.d_model,
        )

        output = self.output_projection(
            attention_output
        )

        return output
