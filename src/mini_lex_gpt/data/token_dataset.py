from __future__ import annotations

import torch
from torch.utils.data import Dataset

class TokenSequenceDataset(
    Dataset[torch.Tensor]
):

    def __init__(
        self,
        token_stream: torch.Tensor,
        sequence_length: int,
        stride: int,
    ) -> None:

        if token_stream.ndim != 1:
            raise ValueError(
                "token_strem must be a 1D tensor."

            )

        if token_stream.dtype != torch.long:
            raise ValueError(
                "sequence_length must be at leasr 2."
            )

        if sequence_length < 2:
            raise ValueError(
                "sequence_length must be at least 2."
            )

        if stride <= 0:
            raise ValueError(
                "stride must be greater than zero."
            )

        if token_stream.numel() < sequence_length:
            raise ValueError(
                "token_stream is shorter than "
                "sequence_length."
            )

        self.token_stream = token_stream 
        self.sequence_length = sequence_length
        self.stride = stride

        self.number_of_sequences = (
            (
                token_stream.numel()
                - sequence_length
            )
            // stride
            + 1
        )

    def __len__(self) -> int:
        return self.number_of_sequences

    def __getitem__(
        self,
        index: int,
    ) -> torch.Tensor:


        if index < 0 or index >= len(self):
            raise IndexError(
                "Dataset index out of range."
            )

        start_index = (
            index * self.stride
        )

        end_index = (
            start_index
            + self.sequence_length
        )

        sequence = self.token_stream[
            start_index:end_index
        ]

        return sequence
