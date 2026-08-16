from __future__ import annotations

import torch
from torch import nn

class RotaryPositionEmbedding(nn.Module):
    """
    
    Aplly Rotary Position Embedding to Query and Key tensors.

    Expected input shape:
        [batch_size, num_heads, sequence:length, head_dimension]

    Output shape:
        The same shape as the input.
    """


    def __init__(
        self,
        head_dimension: int,
        max_sequence_length: int,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()

        if head_dimension <= 0:
            raise ValueError(
                "head_dimension must be greater than zero."
            )
        
        if head_dimension % 2 != 0:
            raise ValueError(
                "head_dimension must be even because RoPE "
                "rotates adjacent pairs of dimensions."
            )

        if max_sequence_length <= 0:
            raise ValueError(
                "max_sequence_length must be greater than zero."
            )

        if base <= 0:
            raise ValueError(
                "base must be greater than zero."
            )

        self.head_dimension = head_dimension
        self.max_sequence_length = max_sequence_length
        self.base = base

        even_dimension_indices = torch.arange(
            start=0,
            end=head_dimension,
            step=2,
            dtype=torch.float32,
        )

        inverse_frequencies = 1.0 / (
           base ** (
               even_dimension_indices
               / head_dimension
           )
        )

        positions = torch.arange(
            max_sequence_length,
            dtype=torch.float32,
        )

        # Shape:
        # [max_sequence_length, head_dimension // 2]
 
        angles = torch.outer(
            positions,
            inverse_frequencies,
        )

        cosine_cache = torch.cos(angles)
        sine_cache = torch.sin(angles)

        self.register_buffer(
            "inverse_frequencies",
            inverse_frequencies,
            persistent=False,
        )

        self.register_buffer(
            "cosine_cache",
            cosine_cache,
            persistent=False,
        )

        self.register_buffer(
            "sine_cache",
            sine_cache,
            persistent=False,
        )
    def _validate_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_offset: int,
    ) -> None:
        """
  
        Validate Query and Key before applying RoPE.
        """

        if query.shape[0] != key.shape[0]:
            raise ValueError(
                "query and key must have the same batch size. "
            )
        if query.shape[-2] != key.shape[-2]:
            raise ValueError(
                "query and key must have the same "
                "sequence length."
            )
        if query.shape[-1] != key.shape[-1]:
            raise ValueError(
                "query and key must have the same  "
                "head dimension."
            )

        if query.ndim != 4:
            raise ValueError(
                "query and key must have shape "
                "[batch, heads, squence, head_dimension]. "
                f"Received {tuple(query.shape)}."
            )

        if query.shape[-1] != self.head_dimension:
            raise ValueError(
                "The final dimension must match "
                f"head_dimension={self.head_dimension}, "
                f"but received {query.shape[-1]}."
            )
       
        if not query.is_floating_point():
            raise TypeError(
                "query and key must use a floating-point dtype."
            )

        if query.dtype != key.dtype:
            raise TypeError(
                "query and key must use the same dtype."
            )

        if query.device != key.device:
            raise ValueError(
                "query and key must be on the same device."
            )

        if position_offset < 0:
            raise ValueError(
                "position_offset cannot be negative."
            )
        sequence_length = query.shape[-2]
        
        required_length = (
            position_offset + sequence_length
        )

        if required_length > self.max_sequence_length:
            raise ValueError(
                "The requested positions exceed the RoPE cache. "
                f"Required length: {required_length}; "
                f"maximum: {self.max_sequence_length}."
            )

    @staticmethod
    def _apply_rotation(
        tensor: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        """
        
        Rotate adjacent pairs in the final dimension.
        
        Input shape:
            [B, H, T, Dh]

        pair shape:
            [B, H, T, Dh // 2, 2]
        """
        original_shape = tensor.shape
        
        tensor_pairs = tensor.reshape(
            *original_shape[:-1],
            original_shape[-1] // 2,
            2,
        )

        x_values = tensor_pairs[..., 0]
        y_values = tensor_pairs[..., 1]

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

        rotated_tensor = rotated_pairs.flatten(
            start_dim=-2,
        )

        return rotated_tensor


    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        
        Apply position-dependent rotations to Query and Key.

        Args:
            query:
                Tensor with shape [B, H, T, Dh].

            key:
               Tensor with shape [B, H, T, Dh].

           position_offset:
               Starting position of the sequence. It is zero
               during ordinary training and may be nonzero
               during cached autoregressive generation.

        Returns:
            rotated_query:
                Tensor with the same shape as query.
            ritated_key:
                Tensor with the same shape as key.
        """

        self._validate_inputs(
            query=query,
            key=key,
            position_offset=position_offset,
        )

        sequence_length = query.shape[-2]

        position_end = (
            position_offset + sequence_length
        )

        cosine = self.cosine_cache[
            position_offset:position_end
        ]

        sine = self.sine_cache[
            position_offset:position_end
        ]

        cosine = cosine.to(
            device=query.device,
            dtype=query.dtype,
        )

        sine = sine.to(
            device=query.device,
            dtype=query.dtype,
        )

        # Before:
        # [T, Dh // 2]
 
        # After:
        # [1, 1, T, Dh // 2]
        #
        # Batch and head dimensions with be broadcast.
        cosine = cosine.unsqueeze(0).unsqueeze(0)
        sine = sine.unsqueeze(0).unsqueeze(0)

        rotated_query = self._apply_rotation(
            tensor=query,
            cosine=cosine,
            sine=sine,
        )

        rotated_key = self._apply_rotation(
            tensor=key,
            cosine=cosine,
            sine=sine,
        )
        
        return rotated_query, rotated_key
