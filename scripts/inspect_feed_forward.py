from __future__ import annotations

import torch

from mini_lex_gpt.config import ModelConfig
from mini_lex_gpt.model.feed_forward import (
    SwiGLUFeedForward,
)

def main() -> None:
    torch.manual_seed(42)
    
    config = ModelConfig()
    
    feed_forward = SwiGLUFeedForward(
        config=config,
    )

    batch_size = 2
    sequence_length = 5

    hidden_states = torch.randn(
       batch_size,
       sequence_length,
       config.d_model,
       dtype=torch.float32,
    )

    output = feed_forward(
        hidden_states
    )
    
    gate_up = feed_forward.gate_up_projection(
        hidden_states
    )
    
    gate, up = gate_up.chunk(
        2,
        dim=-1,
    )

    activated_gate = torch.nn.functional.silu(
        gate
    )

    gated_hidden_states = (
        activated_gate * up
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in feed_forward.parameters()
        if parameter.requires_grad
    )

    print("==================================")
    print(" MiniLexGPT SwiGLU Feed_Forward")
    print("==================================")

    print("\n--- Configuration ---")
    
    print("Model dimension:")
    print(config.d_model)

    print("\nFFN hideen size:")
    print(config.ffn_hidden_size)

    print("\n--- Input ---")

    print("Hidden states shape:")
    print(hidden_states.shape)

    print("\n--- Gate + Up Projection ---")
    
    print("Projection layer:")
    print(feed_forward.gate_up_projection)

    print("\nCombined gate/up shape:")#
    print(gate_up.shape)

    print("\nGate shape:")
    print(gate.shape)

    print("\nUp shape:")
    print(up.shape)

    print("\n--- Activation ---")
    
    print("Activated gate shape:")
    print(activated_gate.shape)

    print("\nGated hidden states shape:")
    print(gated_hidden_states.shape)

    print("\n--- Down Projection ---")
    
    print("Down projection layer:")
    print(feed_forward.down_projection)

    print("\nOutput shape:")
    print(output.shape)

    print("\n--- Parameter check ---")
    
    print("Trainable parameters:")
    print(parameter_count)

    print("\nExpected:")
    print(147_456)

    print(
        "\nParameter count is correct:",
        parameter_count == 147_456,
    )

    print("\n--- Shape Preservation ---")

    print(
        "Input and output shapes match:",
        hidden_states.shape == output.shape,
    )

    print("\n===============================")
    print(" SwiGLU inspection completed")
    print("================================")

if __name__ == "__main__":
    main()
