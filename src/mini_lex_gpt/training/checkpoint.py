from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer
    epoch: int,
    global_step: int,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            opzimizer.state_dict(),

        "epoch":
            epoch,
        "global_step":
            global_step,
    }

    torch.save(
        checkpoint,
        path,
    )
