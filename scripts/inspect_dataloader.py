from __future__ import annotations

import torch
from torch.utils.data import DataLoader
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

    dataset = TokenSequenceDataset(
        token_stream=token_stream,
        sequence_length=4,
        stride=2,
    )

    batch_size = 2

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    print("==========================")
    print(" MiniLexGPT DataLoader")
    print("==========================")

    print("\nDataset length:")
    print(
        len(dataset)
    )

    print("\nDataset length:")
    print(
        len(dataset)
    )

    print("\nBatch size:")
    print(
        batch_size
    )

    print("\nNumber of batches:")
    print(
        len(dataloader)
    )
    
    print("\n--- Batches ---")

    for batch_index, batch in enumerate(
        dataloader
    ):

        print(
            f"\nBatch {batch_index}:"
        )

        print(batch)

        print("Shape:")
        print(
            batch.shape
        )

        print("dtyype:")
        print(
            batch.dtype
        )

    print("\n================================")
    print(" DataLoader inspection completed")
    print("==================================")
if __name__ == "__main__":
    main()
