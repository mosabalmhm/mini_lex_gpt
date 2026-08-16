from __future__ import annotations

import torch
from mini_lex_gpt.config import ModelConfig
from mini_lex_gpt.model.qkv_projection import (
    GroupedQueryProjection,
)

def main() -> None:
    torch.manual_seed(42)

    config = ModelConfig()
    
    projection = GroupedQueryProjection(
        config=config,
    )

    batch_size = 2
    sequence_length = 5
    
    hidden_states = torch.randn(
        batch_size,
        sequence_length,
        config.d_model,
        dtype=torch.float32
    )

    query, key, value = projection(
        hidden_states
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in projection.parameters()
        if parameter.requires_grad
    )

    expected_parameter_count = (
        config.d_model
        * (
            config.query_dimension
            + 2 * config.kv_dimension
        )
    )

    print("===================================")
    print("MiniLexGPT Grouped Query Projection")
    print("===================================")

    print("\n--- Configuration ___")

    print("Model dimension:")
    print(config.d_model)

    print("\nQuery heads:")
    print(config.num_heads)

    print("\nKey/Value heads:")
    print(config.num_kv_heads)

    print("\nHead dimension:")
    print(config.head_dimension)

    print("\nQuery heads per KV head:")
    print(config.query_heads_per_kv_head)

    print("\nQuery dimension_")
    print(config.query_dimension)
 
    print("\nKey/Value dimension:")
    print(config.kv_dimension)
  

    print("\n--- Input ---")

    print("Hidden states shape:")
    print(hidden_states.shape)

    print("\n--- Combined Projection ---")

    print("Linear layer:")
    print(projection.qkv_projection)

    print("\nTotal projection dimension:")
    print(projection.total_projection_dimension)

    print("\n--- Output ---")
    
    print("Query shape:")
    print(query.shape)
    
    print("\nKey shape:")
    print(key.shape)

    print("\nValue shape:")
    print(value.shape)

    print("\n--- Head Grouping ---")
    
    for query_head in range(config.num_heads):
        kv_head = (
            query_head
            // config.query_heads_per_kv_head
        )

        print(
            f"Query head {query_head} "
            f"uses KV head {kv_head}"
        )


    print("\n--- Parameter Check ---")
    
    print("Actual trainable parameters:")
    print(parameter_count)

    print("\nExpected trainable parameters:")
    print(expected_parameter_count)

    print(
        "\nParameter count is correct:",
        parameter_count
        == expected_parameter_count,
    )

    print("\n====================================")
    print(" QKV projection inspection completed")
    print("======================================")

if __name__ == "__main__":
    main()
