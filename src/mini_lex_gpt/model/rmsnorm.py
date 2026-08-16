from __future__ import annotations

import torch
from torch import nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    The layer normalizes values across the final dimension.
    
    Expected input shape:
        [..., dimension]

    Output shape:
        The same shape as the input.
    """

    def __init__(
        self,
        dimension: int,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()

        if dimension <= 0:
            raise ValueError(
                "dimension must be greater than zero."
            )

        if epsilon <= 0:
            raise ValueError(
                "epsilon must be greater than zero."
            )

        self.dimension = dimension
        self.epsilon = epsilon

        # One learnable scaling value for every feature.
        self.weight = nn.Parameter(
            torch.ones(dimension)
        )

    def forward(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """

        Normalize the final dimension using its RMS value.
        """

        if tensor.ndim == 0:
            raise ValueError(
                "tensor must have at least one dimension."
            )

        if tensor.shape[-1] != self.dimension:
            raise ValueError(
                "The final input dimension must match "
                f"dimension={self.dimension}, "
                f"but received {tensor.shape[-1]}."
            )

        if not tensor.is_floating_point():
            raise TypeError(
                "RMSNorm requires a floating-point tensor."
            )

        input_dtype = tensor.dtype

        # float16 and bfloat16 have lower numerical preision.
        # The RMS calculation is therefore performed in folat32.
        if tensor.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            tensor_for_norm = tensor.float()
        else:
            tensor_for_norm = tensor
        
        mean_square = tensor_for_norm.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        inverse_rms = torch.rsqrt(
            mean_square + self.epsilon
        )

        normalized_tensor = (
            tensor_for_norm * inverse_rms
        )

        normalized_tensor = normalized_tensor.to(
            dtype=input_dtype
        )

        return normalized_tensor * self.weight
