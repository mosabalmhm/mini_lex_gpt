from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from mini_lex_gpt.config import ModelConfig
from mini_lex_gpt.data.token_dataset import (
    TokenSequenceDataset,
)
from mini_lex_gpt.model.language_model import (
    MiniLexGPT,
)
from mini_lex_gpt.training.optimizer import (
    build_adamw_optimizer,
)
from mini_lex_gpt.training.train_step import (
    train_step,
)

def main() -> None:
    
    torch.manual_seed(42)

    config = ModelConfig()
    
    model = MiniLexGPT(
        config=config,
    )

    optimizer = build_adamw_optimizer(
        model=model,
        learning_rate=1e-3,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
    )

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

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )

    number_of_epochs = 10
    
    global_step = 0

    print("==========================")
    print(" MiniLexGPT Training Loop")
    print("===========================")

    print("\nDataset samples:")
    print(len(dataset))

    print("\nBatches per epoch:")
    print(len(dataloader))

    print("\nNumber of epochs:")
    print(number_of_epochs)

    expected_total_steps = (
        number_of_epochs
        * len(dataloader)
    )

    print("\nExpected optimizer steps:")
    print(expected_total_steps)

    for epoch in range(
        1,
        number_of_epochs + 1,
    ):

        epoch_loss_sum = 0.0
        epoch_gradient_norm_sum = 0.0

        print(
            f"\n--- Epoch {epoch:02d} ---"
        )

        for batch_index, batch in enumerate(
            dataloader
        ):
            loss, gradient_norm = train_step(
                model=model,
                optimizer=optimizer,
                token_ids=batch,
                max_gradient_norm=1.0,
            )

            global_step += 1
            
            epoch_loss_sum += (
                loss.item()
            )

            epoch_gradient_norm_sum += (
                gradient_norm.item()
            )

            print(
                f"step={global_step:02d} | "
                f"batch={batch_index} | "
                f"loss={loss.item():.6f} | "
                f"grad_norm="
                f"{gradient_norm.item():.6f}"
            )

        mean_epoch_loss = (
            epoch_loss_sum
            / len(dataloader)
        )

        mean_epoch_gradient_norm = (
            epoch_gradient_norm_sum
            / len(dataloader)
        )

        print(
            f"Epoch {epoch:02d} mean | "
            f"loss={mean_epoch_loss:.6f} | "
            f"grad_norm="
            f"{mean_epoch_gradient_norm:.6f}"
        )

    print("\n--- Final checks ---")

    print("\nActual optimizer steps:")
    print(global_step)

    print("\nExpected optimizer steps:")
    print(expected_total_steps)

    print("\nStep count is correct:")
    print(
        global_step
        == expected_total_steps
    )

    print("\n==========================")
    print(" Training loop completed")
    print("============================")

if __name__ == "__main__":
    main()
