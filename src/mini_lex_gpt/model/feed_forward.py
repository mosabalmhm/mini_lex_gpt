from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mini_lex_gpt.config import ModelConfig

class SwiGLUFeedForward(nn.Module):
    """

    SwiGLU feed-forward network.

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
        self.hidden_size = config.ffn_hidden_size

        self.gate_up_projection = nn.Linear(
            in_features=self.d_model,
            out_features=2 * self.hidden_size,
            bias=config.use_bias,
        )

        self.down_projection = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.d_model,
            bias=config.use_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

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
                "hidden_states must use a floating-point dtype."
            )

        gate_up = self.gate_up_projection(
            hidden_states
        )

        gate, up = gate_up.chunk(
            2,
            dim=-1,
        )

        activated_gate = F.silu(
            gate
        )

        gated_hidden_states = (
            activated_gate * up
        )

        output = self.down_projection(
            gated_hidden_states
        )
        
        return output
