from __future__ import annotations

import torch

from mini_lex_gpt.data.token_dataset import (
    TokenSequenceDataset,
)

def main() -> None:
    
    token_stream = torch.tensor(
        [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
        ],
        dtype=torch.long,
    )

    sequence_length = 4
    stride = 2

    dataset = TokenSequenceDataset(
        token_stream=token_stream,
        sequence_length=sequence_length,
        stride=stride,
    )

    expected_number_of_sequences = (
        (
            token_stream.numel()
            - sequence_length
        )
        // stride
        + 1
    )

    print("=============================")
    print(" MiniLexGPT Token Dataset")
    print("=============================")

    print("\nToken stream:")
    print(token_stream)

    print("\nToken count:")
    print(token_stream.numel())

    print("\nSequence length:")
    print(sequence_length)

    print("\nStride:")
    print(stride)

    print("\nExpected number of sequences:")
    print(expected_number_of_sequences)

    print("\nDataset length:")
    print(len(dataset))

    print("\n--- sequences ---")
    
    for index in range(
        len(dataset)
    ):
        sequence = dataset[index]

        print(
            f"{index}: "
            f"{sequence.tolist()}"
        )

    print("\nFirst sequence shape:")
    print(
        dataset[0].shape
    )

    print("\n=============================")
    print(" Dataset inspection comleted")
    print("==================================")

if __name__ == "__main__":
    main()
