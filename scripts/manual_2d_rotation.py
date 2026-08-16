from __future__ import annotations

import math
import torch

def rotate_2d(
    vector: torch.Tensor,
    angle_radians: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    
    Rotate a two-dimensional vector by a given angle.

    Args:
        vector:
            Tensor with shape [2].
        angle_radians:
            Rotation angle measured in radians.

    Returns:
        A tuple containing:
        - the rotated vector
        - the rotation matrix
    """

    if vector.shape != (2,):
        raise TypeError(
            "vector must have shape[2],"
            f"but received {tuple(vector.shape)}."
        )

    if not vector.is_floating_point():
        raise TypeError(
            "vector must use a floating-point data type."
        )


    angle = torch.tensor(
        angle_radians,
        dtype=vector.dtype,
        device=vector.device,
    )

    cosine = torch.cos(angle)
    sine = torch.sin(angle)

    rotation_matrix = torch.stack(
        [
            torch.stack([cosine, -sine]),
            torch.stack([sine, cosine]),
        ]
    )

    rotated_vector = rotation_matrix @ vector

    return rotated_vector, rotation_matrix

def clean_small_values(
    tensor: torch.Tensor,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    """

    Replace tiny floating-point values with exact zeros.
    """

    return torch.where(
        tensor.abs() < tolerance,
        torch.zeros_like(tensor),
        tensor,
    )

def main() -> None:
    token_vector = torch.tensor(
        [1.0, 0.0],
        dtype=torch.float32,
    )

    angle_per_position = math.pi / 2
    
    orginal_length = torch.linalg.vector_norm(
       token_vector
    )

    print("================================")
    print(" Manual Two-Dimensional Rotation")
    print("===================================")

    print("\nOrginal token vector:")
    print(token_vector)

    print("\nOrginal vector length:")
    print(orginal_length.item())

    print("\nAngle per position:")
    print("Radians:", angle_per_position)
    print("Degrees:", math.degrees(angle_per_position))

    print("\n--- Position Rotations ---")
    
    for position in range(4):
        angle = position * angle_per_position

        rotated_vector, rotation_matrix = rotate_2d(
            vector=token_vector,
            angle_radians=angle,
        )

        rotated_vector = clean_small_values(
            rotated_vector
        )

        rotation_matrix = clean_small_values(
            rotation_matrix
        )

        rotated_length = torch.linalg.vector_norm(
            rotated_vector
        )

        print(f"\nposition {position}")
        print(
            "Angle:",
            f"{angle:.2f} radians",
            f"({math.degrees(angle):.1f} degrees)",
        )

        print("Rotation matrix:")
        print(rotation_matrix)
        
        print("Rotated vector:")
        print(rotated_vector)

        print("Rotated vector length:")
        print(rotated_length.item())

    print("\n==============================")
    print(" Rotation experiment completed")
    print("================================")

if __name__ == "__main__":
    main()
