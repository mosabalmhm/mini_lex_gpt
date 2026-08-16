from __future__ import annotations

import torch
from torch import nn

from mini_lex_gpt.config import ModelConfig

class GroupedQueryProjection(nn.Module):
    """
    Project hidden states into Query , Key, and Value tensors
    for Grouped-Query Attention.

    Input shape:
        [batch_size, sequence_length, d_model]

    Query output:
        [batch_size, num_query_heads,
         sequence_length, head_dimension]

    Key and Value outputs:
        [batch_size, num_kv_dimension]
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
        
        self.query_dimension = (
            self.num_query_heads
            * self.head_dimension
        )

        self.kv_dimension = (
            self.num_kv_heads
            * self.head_dimension
        )

        self.total_projection_dimension = (
            self.query_dimension
            + self.kv_dimension
            + self.kv_dimension
        )

        self.qkv_projection = nn.Linear(
            in_features=self.d_model,
            out_features=self.total_projection_dimension,
            bias=config.use_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Create Query, Key, and value tensors.
        """

        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape "
                "[batch, sequence, d_model], "
                f"but received {tuple(hidden_states.shape)}."
            )

        if hidden_states.shape[-1] != self.d_model:
            raise ValueError(
                "The final hidden-states dimension must equal "
                f"d_model={self.d_model}, but received "
                f"{hidden_states.shape[-1]}."
            )
        if not hidden_states.is_floating_point():
            raise TypeError(
                "hidden_states must use a floating-point dtype."
            )

        batch_size = hidden_states.shape[0]
        sequence_length = hidden_states.shape[1]

        combined_qkv = self.qkv_projection(
            hidden_states
        )

        query, key, value = torch.split(
            combined_qkv,
            [
                self.query_dimension,
                self.kv_dimension,
                self.kv_dimension,
            ],
            dim=-1,
        )
        
        query = query.reshape(
            batch_size,
            sequence_length,
            self.num_query_heads,
            self.head_dimension,
        )

        key = key.reshape(
            batch_size,
            sequence_length,
            self.num_kv_heads,
            self.head_dimension,
        )

        value = value.reshape(
            batch_size,
            sequence_length,
            self.num_kv_heads,
            self.head_dimension,
        )


        # Before:
        # [B, T, H, Dh]
        # 
        # After:
        # [B, H, T, Dh]

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        return query, key, value
