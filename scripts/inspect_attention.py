from __future__ import annotations

import torch
from mini_lex_gpt.config import ModelConfig
from mini_lex_gpt.model.attention import (
    CausalGroupedQueryAttention,
)

def main() -> None:
    torch.manual_seed(42)
    
    config = ModelConfig()
    
    attention = CausalGroupedQueryAttention(
        config=config,
    )

    attention.eval()

    batch_size = 2
    sequence_length = 5

    hidden_states = torch.randn(
        batch_size,
        sequence_length,
        config.d_model,
        dtype=torch.float32,
    )

    output = attention(
        hidden_states
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in attention.parameters()
        if parameter.requires_grad
    )

    print("==================================")
    print(" MiniLexGPT Causal GQA")
    print("====================================")

    print("\n--- Input ---")
    
    print("Hidden states shape:")
    print(hidden_states.shape)

    print("\n--- Attention Configuration ---")
    
    print("Query heads:")
    print(config.num_heads)

    print("\nKV heads:")
    print(config.num_kv_heads)

    print("\nHead dimension:")
    print(config.head_dimension)

    print("\nQuery heads per KV head:")
    print(config.query_heads_per_kv_head)

    print("\n--- Output ---")
    
    print("Attention output shape:")
    print(output.shape)

    print("\n--- Parameter Count ---")
    print("Trainable attention parameters:")
    print(parameter_count)

    print("\nExpected:")
    print(49_152)
    
    print(
        "\nParameter count is correct:",
        parameter_count == 49_152,
    )

    print("\n--- Causality Experiment ---")

    first_input = torch.randn(
        1,
        sequence_length,
        config.d_model,
    )

    second_input = first_input.clone()

    second_input[:, -1, :] = (
        second_input[:, -1, :] + 100.0
    )

    first_output = attention(first_input)
    second_output = attention(second_input)

    earlier_positions_equal = torch.allclose(
        first_output[:, :-1],
        second_output[:, :-1],
        atol=1e-5,
    )

    final_position_changed = not torch.allclose(
        first_output[:, -1],
        second_output[:, -1],
        atol=1e-5,
    )
    
    print(
        "Changing the future token leaves "
        "earlier outputs unchanged",
        earlier_positions_equal,
    ) 

    print(
        "The changed token affects its own output:",
        final_position_changed,
    )  

    print("\n==============================")
    print(" Attention inspection completed")
    print("==================================")

if __name__ == "__main__":
    main()
