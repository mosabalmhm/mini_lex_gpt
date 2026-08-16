from __future__ import annotations

import torch

from mini_lex_gpt.model.rope  import (
    RotaryPositionEmbedding,
)

def main() -> None:
    torch.manual_seed(42)

    batch_size = 1
    num_heads = 2
    sequence_length = 4
    head_dimension = 8
    max_sequence_length = 8

    rope = RotaryPositionEmbedding(
        head_dimension=head_dimension,
        max_sequence_length=max_sequence_length,
        base=10_000.0,
    )

    base_vector = torch.tensor(
        [
            1.0, 0.0,
            1.0, 0.0,
            1.0, 0.0,
            1.0, 0.0,
        ],
        dtype=torch.float32,
    )

    query = base_vector.reshape(
        1, 1, 1, head_dimension
    ).repeat(
        batch_size,
        num_heads,
        sequence_length,
        1,
    )

    key = query.clone()

    rotated_query, rotated_key = rope(
        query=query,
        key=key,
    )

    original_lengths = torch.linalg.vector_norm(
        query,
        dim=-1,
    )

    rotated_lengths = torch.linalg.vector_norm(
        rotated_query,
        dim=-1,
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in rope.parameters()
        if parameter.requires_grad
    )

    print("===================================")
    print(" MiniLexGPT Rotary Position Embedding")
    print("=====================================")

    print("\n--- Layer Configuration ---")

    print("Head dimension:")
    print(rope.head_dimension)
    
    print("Maximum sequence length:")
    print(rope.max_sequence_length)
    
    print("RoPE base:")
    print(rope.base)

    print("\n--- Cache shapes ---")

    print("Inverse frequencies:")
    print(rope.inverse_frequencies)

    print("\nCosine-cache shape:")
    print(rope.cosine_cache.shape)

    print("\nSine-cache shape:")
    print(rope.sine_cache.shape)

    print("\n--- Output shapes ---")
    
    print("Rotated Query shape:")
    print(rotated_query.shape)

    print("\nRotated key shape:")
    print(rotated_key.shape)

    print("\n--- Position Examples ---")
    
    for position in range(sequence_length):
        print(f"\nPosition {position}")

        print("Original vector:")
        print(query[0, 0, position])

        print("Rotated vector:")
        print(rotated_query[0, 0, position])

        print("Original length:")
        print(original_lengths[0, 0, position].item())

        print("Rotated length:")
        print(rotated_lengths[0, 0, position].item())

    print("\n--- Correctness Checks ---")
     
    position_zero_unchanged = torch.allclose(
         query[:, :, 0],
         rotated_query[:, :, 0],
         atol=1e-6,
    )

    query_and_key_equal = torch.allclose(
        rotated_query,
         rotated_key,
         atol=1e-6,
    )
    
    lengths_preserved = torch.allclose(
         original_lengths,
         rotated_lengths,
         atol=1e-6,
 
    )

    heads_receive_same_rotation = torch.allclose(
        rotated_query[:, 0],
        rotated_query[:, 1],
        atol=1e-6,
    )
     
    print(
        "Position zero remains unchanged:",
        position_zero_unchanged,
    )

    print(
        "Equal Query and key remain equal:",
        query_and_key_equal,
    )
     
    print(
        "Vector length are preserved:",
        lengths_preserved,
    )

    print(
        "Identical heads receive identical rotations:",
        heads_receive_same_rotation,
    )

    print("\n--- Parameters and Buffers ---")
     
    print(
        "Trainable parameter count:",
        trainable_parameter_count,
    )

    print("\nRegistered buffers:")
    for buffer_name, buffer in rope.named_buffers():
        print(
            f"{buffer_name:24s}: "
            f"shape={tuple(buffer.shape)}"
        )

    print("\nState dictionary keys:")
    print(list(rope.state_dict().keys()))

    print("\n================================")
    print(" RoPE inspection completed")
    print("=================================")

if __name__ == "__main__":
    main()
