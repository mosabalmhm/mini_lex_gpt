from __future__ import annotations

import torch

from mini_lex_gpt.config import ModelConfig
from mini_lex_gpt.model.transformer_block import(
    TransformerBlock,
)

def main() -> None:
    torch.manual_seed(42)

    config = ModelConfig()

    block = TransformerBlock(
        config=config,
    )

    block.eval()

    batch_size = 2
    sequence_length = 5

    hidden_states = torch.randn(
        batch_size,
        sequence_length,
        config.d_model,
        dtype=torch.float32,
    )

    output = block(
        hidden_states
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in block.parameters()
        if parameter.requires_grad
    )

    output_changed = not torch.allclose(
        hidden_states,
        output,
        atol=1e-6,
    )

    print("=============================")
    print(" MiniLexGPT Transormer Block")
    print("=============================")

    print("\n--- Configuration ---")
    
    print("Model dimension:")
    print(config.d_model)

    print("\nAttention heads:")
    print(config.num_heads)


    print("\nKV heads:")
    print(config.num_kv_heads)

    print("\nFFN hidden size:")
    print(config.ffn_hidden_size)

    print("\n--- Input ---")

    print("Input shape:")
    print(hidden_states.shape)

    print("\n--- Output ---")

    print("Output shape:")
    print(output.shape)

    print("\n--- Shape Check ---")
    
    print(
        "Input and output shapes match:",
        hidden_states.shape == output.shape,
    )

    print("--- Transformation Check ---")
    
    print(
        "Block changed the hidden-state Values:",
        output_changed,
    )
   
    print("\n--- Parameter Check ---")

    print("Trainable block parameters:")
    print(parameter_count)

    print("\nExpected:")
    print(196_864)

    print(
        "\nParameter count is correct:",
        parameter_count == 196_864,
    )
    
    print("\n--- Submodule Parameters ---")

    attention_parameters = sum(
        parameter.numel()
        for parameter in block.attention.parameters()
        if parameter.requires_grad
    )


    feed_forward_parameters = sum(
        parameter.numel()
        for parameter in block.feed_forward.parameters()
        if parameter.requires_grad
    )

    attention_norm_parameters = sum(
        parameter.numel()
        for parameter in block.attention_norm.parameters()
        if parameter.requires_grad
    )
 
    feed_forward_norm_parameters = sum(
        parameter.numel()
        for parameter in block.feed_forward_norm.parameters()
        if parameter.requires_grad
    )

    print("Attention:")
    print(attention_parameters)

    print("\nFeed-forward:")
    print(feed_forward_parameters)

    print("\nAttention RMSNorm:")
    print(attention_norm_parameters)

    print("\nFeed-forward RMSNorm:")
    print(feed_forward_norm_parameters)

    print("\n---------------------------------------")
    print(" Tranformer block inspection completed")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()
