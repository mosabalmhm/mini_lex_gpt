from __future__ import annotations

import torch

def create_rotation_matrix(
    angle_radians:float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """

    Create a two-dimensional rotation matrix.

    output shape:
        [2, 2]
    """

    angle = torch.as_tensor(
        angle_radians,
        dtype=dtype,
        device=device,
    )

    cosine = torch.cos(angle)
    sine = torch.sin(angle)

    first_row = torch.stack(
        [cosine, -sine]
    )

    second_row = torch.stack(
        [sine, cosine]
    )

    rotation_matrix = torch.stack(
        [first_row, second_row]
    )

    return rotation_matrix

def rotate_vector(
    vector: torch.Tensor,
    angle_radians: float,
) -> torch.Tensor:
    """
    Rotate a vector with shape [2].
    """
    if vector.shape != (2,):
        raise ValueError(
            "vector must have shape [2], "
            f"but received {tuple(vector.shape)}."
        )

    if not vector.is_floating_point():
        raise TypeError(
            "vector must use a floating-point dtype."
        )

    rotation_matrix = create_rotation_matrix(
        angle_radians,
        dtype=vector.dtype,
        device=vector.device,
    )

    return rotation_matrix @ vector

def calculate_positional_score(
    query: torch.Tensor,
    key: torch.Tensor,
    query_position: int,
    key_position: int,
    frequency: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    
    Rotate Query and key according to their  positions,
    then calculate their dot-product score.

    Returns:
        rotated_query
        rotated_key
        attention_score
    """

    if query_position < 0 or key_position < 0:
        raise ValueError(
            "Position cannot be negative."
        )

    if frequency <= 0:
        raise ValueError(
            "frequency must be greater than zero."
        )

    query_angle = query_position * frequency
    key_angle = key_position * frequency

    rotated_query = rotate_vector(
        query,
        query_angle,
    )

    rotated_key = rotate_vector(
        key,
        key_angle,
    )

    attention_score = torch.dot(
        rotated_query,
        rotated_key,
    )

    return(
        rotated_query,
        rotated_key,
        attention_score,
    )

def main() -> None:
    query = torch.tensor(
        [1.0, 0.0],
        dtype=torch.float32,
    )

    key = torch.tensor(
        [1.0, 0.0],
        dtype=torch.float32,
    )

    frequency = 0.5

    position_cases = [
        {
            "name": "Case A",
            "query_position": 2,
            "key_position": 5,
        },
        { 
            "name": "Case B",
            "query_position": 7,
            "key_position": 10,
        },
        {
            "name": "Case C",
            "query_position": 2,
            "key_position": 4,
        },
    ]

    scores: dict[str, torch.Tensor] = {}

    print("=====================================")
    print(" RoPE Relative Position Experiment")
    print("=====================================")

    print("\nBase Query:")
    print(query)
   
    print("\nBase Key:")
    print(key)

    print("\nFrequency:")
    print(frequency, "radians per position")

 
    print("\n--- Position Cases ---")
    
    for case in position_cases:
        name = case["name"]
        query_position = case["query_position"]
        key_position = case["key_position"]

        relative_distance = (
            key_position - query_position
        )

        query_angle = query_position * frequency
        key_angle = key_position * frequency
        
        (
            rotated_query,
            rotated_key,
            attention_score,
        ) = calculate_positional_score(
            query=query,
            key=key,
            query_position=query_position,
            key_position=key_position,
            frequency=frequency,
        )

        expected_score = torch.cos(
            torch.tensor(
                relative_distance * frequency,
                dtype=query.dtype,
            )
        )

        scores[name] = attention_score

        print(f"\n{name}")

        print("Query positions:")
        print(query_position)

        print("key_position:")
        print(key_position)

        print("Relative distance:")
        print(relative_distance)


        print("Query angle:")
        print(query_angle)

        print("Key angle:")
        print(key_angle)

        print("Rotated Query:")
        print(rotated_query)

        print("Rotated key")
        print(rotated_key)

        print("Dot-product score:")
        print(attention_score.item())

        print("Expected relative-position score:")
        print(expected_score.item())
 
        print(
            "Calculated and expected scores agree:",
            torch.allclose(
                attention_score,
                expected_score,
                expected_score,
                atol=1e-6,
            ),
        )
        
    print("\n--- Comparison ---")
      
    same_relative_distance = torch.allclose(
        scores["Case A"],
        scores["Case B"],
        atol=1e-6,
    )

    different_relative_distance = not torch.allclose(
        scores["Case A"],
        scores["Case C"],
        atol=1e-6,
    )  

    
    print(
        "Cases A and B have the same scores:",
        same_relative_distance,
    )
    print(
        "Cases A and C have different scores:",
        different_relative_distance,
    )

    print("\n========================================")
    print(" Relative position experiment completed")
    print("==========================================")

if __name__ == "__main__":
    main()
