from __future__ import annotations

import torch

def calculate_inverse_frequencies(
    dimension: int,
    rope_base: float,
) -> torch.Tensor:
    """
    
    Calculate one RoPE frequency for every pair of dimensions.
    for dimension = 8, the dimension indices are:
        0, 2, 4, 6
    output shape:
        [dimension // 2]
    """

    if dimension <= 0:
        raise ValueError(
            "dimension must be greater than zero."
        )

    if dimension % 2 != 0:
        raise ValueError(
            "dimension must be even because RoPE"
            "rotates pairs of values."
        )
  
    if rope_base <= 0:
        raise ValueError(
            "rope_base must be greater than zero."
        )

    even_dimension_indices = torch.arange(
        start=0,
        end=dimension,
        step=2,
        dtype=torch.float32,
    )

    base_tensor = torch.tensor(
        rope_base,
        dtype=torch.float32,
    )

    inverse_frequencies = 1.0 / torch.pow(
       base_tensor,
       even_dimension_indices / dimension,
    )

    return inverse_frequencies

def rotate_dimension_pairs(
    vector: torch.Tensor,
    position: int,
    inverse_frequencies: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """

    Rotate adjacent pairs of dimension.

    Input vector:
        [x0, y0, x1, y1,...]

    output:
        The rotated vector and the angles used.
    """

    if vector.ndim != 1:
        raise ValueError(
            "vector must be one-dimensional."
        )

    if vector.numel() % 2 != 0:
        raise ValueError(
            "vector must contain an even number of values."
        )

    expected_frequency_count = vector.numel() // 2

    if inverse_frequencies.numel() != expected_frequency_count:
        raise ValueError(
            "The number of frequencies must equal "
            "the number of dimension pairs."
        )

    if position < 0:
        raise ValueError(
            "Position cannot be negative."
        )

    # Convert:
    # [8] -> [4, 2]
    vector_pairs = vector.reshape(-1, 2)

    angles = position * inverse_frequencies

    cosine = torch.cos(angles)
    sine = torch.sin(angles)

    x_values = vector_pairs[:, 0]
    y_values = vector_pairs[:, 1]

    rotated_x = (
        x_values * cosine
        - y_values * sine
    )
    
    rotated_y = (
        x_values * sine
        + y_values * cosine
    )

    rotated_pairs = torch.stack(
        [rotated_x, rotated_y],
        dim=-1,
    )

    # Convert:
    # [4, 2] -> [8]
    
    rotated_vector = rotated_pairs.flatten()

    return rotated_vector, angles


def main() -> None:
    dimension = 8
    rope_base = 10_000.0
 
    token_vector = torch.tensor(
        [
           1.0, 0.0,
           1.0, 0.0,
           1.0, 0.0,
           1.0, 0.0,
        ],
        dtype=torch.float32,
    )

    inverse_frequencies = calculate_inverse_frequencies(
        dimension=dimension,
        rope_base=rope_base,
    )

    original_length = torch.linalg.vector_norm(
        token_vector
    )

    print("===============================")
    print(" Manual RoPE Dimension Pairs")
    print("===============================")

    print("\nOriginal vector:")
    print(token_vector)

    print("\nOriginal vector shape:")
    print(token_vector.shape)

    print("\nOriginal vector length:")
    print(original_length.item())

    print("\n--- Iverse Frequencies ---")
    
    for pair_index, frequency in enumerate(
        inverse_frequencies
    ):
        first_dimension = pair_index * 2
        second_dimension = first_dimension + 1

        print(
            f"Pair {pair_index} "
            f"(dimension {first_dimension}, "
            f"{second_dimension}): "
            f"{frequency.item():.6f}"
        )

    print("\n--- Position Rotations ---")

    for position in range(4):
        rotated_vector, angles = rotate_dimension_pairs(
            vector=token_vector,
            position=position,
            inverse_frequencies=inverse_frequencies
        )

        rotated_length = torch.linalg.vector_norm(
            rotated_vector
        )


        print(f"\nPosition {position}")
        
        for pair_index in range(
            inverse_frequencies.numel()
        ):
            start = pair_index * 2
            end = start + 2

            pair_vector = rotated_vector[start:end]

 
            print(
                f"Pair {pair_index}:"
                f"angle={angles[pair_index].item():.6f},"
                f"vector={pair_vector.tolist()}"
            )

        print("Complete rotated vector:")
        print(rotated_vector)

        print("Rotated vector length:")
        print(rotated_length.item())


        length_are_equal = torch.allclose(
            original_length,
            rotated_length,
            atol=1e-6,
        )

        print(
            "Original and rotated length are equal:",
            length_are_equal,
        )

    print("\n=======================================")
    print(" Manual RoPE pair experiment completed")
    print("=========================================")

if __name__ == "__main__":
    main()
