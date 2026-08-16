from __future__ import annotations

import torch

from mini_lex_gpt.config import ModelConfig
from mini_lex_gpt.model.embedding import TokenEmbedding

def main() -> None:
    """

    Demostrate that token embeddings represent token identity,
    but do not explicitly encode token postion.
    """

    torch.manual_seed(42)

    config = ModelConfig(
        vocab_size=16,
        context_length=8,
        d_model=8,
        num_layers=1,
        num_heads=4,
        ffn_hidden_size=16,
        dropout=0.0,
    )

    embedding = TokenEmbedding(config)

    # The same tokens appear in both sequences,
    # but their order is reversed.

    sequence_a = torch.tensor(
        [[1, 2, 3]],
        dtype=torch.long,
    )

    sequence_b = torch.tensor(
        [[3, 2, 1]],
        dtype=torch.long,
    )

    embedded_a = embedding(sequence_a)
    embedded_b = embedding(sequence_b)

    print("================================")
    print(" Why Position Information Matters")
    print("===================================")

    print("\n--- Token Sequences ---")
    
    print("Sequence A:")
    print(sequence_a)

    print("\nSequence B:")
    print(sequence_b)

    print("\n--- Shapes ---")

    print("Sequence A shape:")
    print(sequence_a.shape)

    print("\nEmbedded A shape:")
    print(embedded_a.shape)

    print("\n--- Token Identity check ---")

    # Token 1 is at postion 0 in sequence A
    # and position 2 in sequence B.
    token_one_same_vector = torch.allclose(
        embedded_a[0, 0],
        embedded_b[0, 2],
    )
    print(
        "Token ID 1 has same embedding ",
        "at different positions:",
        token_one_same_vector,
    )

    # Token 2 is in the middle of both sequences.
    token_two_same_vector = torch.allclose(
        embedded_a[0, 1],
        embedded_b[0, 1],
    )
    
    print(
        "Token ID 2 has same embedding:",
        token_two_same_vector,
    )
    
    print("\n--- Complete Sequence Comparison ---")
    
    complete_sequences_equal = torch.allclose(
        embedded_a,
        embedded_b,
    )

    print(
        "Complete embedded sequences are equal:",
        complete_sequences_equal,
    )

    print("\nFirst vector in sequence A:")
    print(embedded_a[0, 0])

    print("\nLast vector in sequence B:")
    print(embedded_b[0, 2])

    print("\n--- Order Removal Experiment ---")

    # Sum across the sequence dimension 
    # This removes information about where each token appeared.

    sequence_a_sum = embedded_a.sum(dim=1)
    sequence_b_sum = embedded_b.sum(dim=1)

    print("Sum of Sequence A embeddings:")
    print(sequence_a_sum)

    print("\nSum of Sequence B embeddings:")
    print(sequence_b_sum)

    sums_are_equal = torch.allclose(
        sequence_a_sum,
        sequence_b_sum,
    )

    print(
        "\nThe sums are equal after order is removed:",
        sums_are_equal,

    )    

    print("\n=============================")
    print(" Postion experiment completed")
    print("================================")

if __name__ == "__main__":
    main()
