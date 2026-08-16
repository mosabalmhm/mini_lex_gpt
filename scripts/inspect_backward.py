from __future__ import annotations

import torch
import torch.nn.functional as F

from mini_lex_gpt.config import ModelConfig
from mini_lex_gpt.model.language_model import MiniLexGPT

def main() -> None:
    torch.manual_seed(42)

    config = ModelConfig()

    model = MiniLexGPT(
        config=config,
    )

    model.train()

    token_ids = torch.tensor(
        [
            [1, 25, 90, 500, 3],
            [1, 70, 90, 800, 3],
        ],
        dtype=torch.long,
    )

    logits = model(
        token_ids
    )

    prediction_logits = logits[
        :, :-1, :
    ]

    targets = token_ids[
        :, 1:
    ]

    flat_prediction_logits = (
        prediction_logits.reshape(
            -1,
            config.vocab_size,
        )
    )

    flat_targets = targets.reshape(
        -1
    )

    loss = F.cross_entropy(
        flat_prediction_logits,
        flat_targets,
    )

    qkv_weight = (
        model.blocks[0]
        .attention
        .qkv
        .qkv_projection
        .weight
    )

    embedding_weight = (
        model.token_embedding
        .embedding
        .weight
    )

    print("===================================")
    print(" MiniLexGPT Backpropagation")
    print("==============================")

    print("Loss:")
    print(loss)

    print("\nQKV hradient before backward:")
    print(embedding_weight.grad)

    loss.backward()

    print("\n--- After Backward ---")
    
    print("QKV gradient exists:")
    print(
        qkv_weight.grad is not None
    )

    print("QKV gradient shape:")
    print(
        qkv_weight.grad.shape
    )
    
    

    print("\nQKV gradient mean:")
    print(
        qkv_weight.grad.mean().item()
    )

    print("\nQKV gradient standard deviation:")
    print(
        qkv_weight.grad.std().item()
    )

    print("\nQKV gradient norm:")
    print(
        qkv_weight.grad.norm().item()
    )
    
    print("\nEmbedding gradient exists:")
    print(
        embedding_weight.grad is not None
    )

    print("\nEmbedding gradient shape:")
    print(
        embedding_weight.grad.shape
    )

    print("")

    print("\nEmbedding gradient norm:")
    print(
        embedding_weight.grad.norm().item()
    )


    print("\nAll QKV gradient finite:")
    print(
        torch.isfinite(
            qkv_weight.grad
        ).all().item()
    )

    print("\nAll embedding gradient finite:")
    print(
        torch.isfinite(
            embedding_weight.grad
        ).all().item()
    )

    print("\n========================================")
    print(" Backpropagation inspection completed")
    print("==========================================")

if __name__ == "__main__":
    main()
