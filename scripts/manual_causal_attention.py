from __future__ import annotations

import math

import torch
import torch.nn.functional as F
def expand_kv_heads(
    tensor: torch.TTensor,
    num_query_heads: int,
) -> torch.Tensor:
    """
    
    Repeat Key or Value heads so that their head count
    matchs the number of Query heads.

    Input shape:
        [B, HKv, T, Dh]
    
    Output shape:
        [B, Hq, T, Dh]
    """

    if tensor.ndim != 4:
        raise ValueError(
            "tensor must have shape [B, HKv, T, Dh]."
        )

    num_kv_heads = tensor.shape[1]

    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "num Query heads must be dvisible by "
            "num_kv_heads."
        )

    repetitions = num_query_heads // num_kv_heads

    return tensor.repeat_interleave(
        repetitions,
        dim=1,
    )

def main() -> None:
    # Shape:
    # [batch=1, query_heads=2, sequence=3, head_dimension=2]
    query = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                [
                    [1.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ]
        ],
        dtype=torch.float32,
    )
    
    # One shared Key head.
    # Shape: [1,1, 3,2]
    key = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    #One shared Value head.
    # Shape: [1, 1, 3, 2]
    value = torch.tensor(
        [
            [
                [
                    [10.0, 0.0],
                    [0.0, 20.0],
                    [30.0, 30.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    num_query_heads = query.shape[1]
    sequence_length = query.shape[-2]
    head_dimension =query.shape[-1]

    expanded_key = expand_kv_heads(
        tensor=key,
        num_query_heads=num_query_heads,
    )

    expanded_value = expand_kv_heads(
        tensor=value,
        num_query_heads=num_query_heads,
    )

    # Q @ KT 
    raw_scores = query @ expanded_key.transpose(
        -2,
        -1,
    )

    scale = math.sqrt(head_dimension)

    scaled_scores = raw_scores / scale
    
    causal_mask = torch.tril(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
        )
    )

    masked_scores = scaled_scores.masked_fill(
        ~causal_mask,
        float("-inf"),
    )

    attention_weights = torch.softmax(
        masked_scores,
        dim=-1,
    )

    manual_output = (
        attention_weights @ expanded_value
    )

    sdpa_output =F.scaled_dot_product_attention(
        query=query,
        key=expanded_key,
        value=expanded_value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
    )

    expanded_future_mask = (
        ~causal_mask
    ).unsqueeze(0).unsqueeze(0).expand_as(
        attention_weights
    )

    future_weights = attention_weights[
        expanded_future_mask
    ]

    print("==========================")
    print(" Manual Causal Attention")
    print("===========================")

    print("\n--- Shapes ---")

    print("Query:")
    print(query.shape)

    print("\nOriginal key:")
    print(key.shape)

    print("\nExpanded Key:")
    print(expanded_key.shape)

    print("\nExpanded value:")
    print(expanded_value.shape)

    print("\n--- Causal Mask ---")
    print(causal_mask)

    for head_index in range(num_query_heads):
        print(f"\n==============================")
        print(f"Query Head {head_index}")
        print("====================== ========")

        print("\nRaw QK**T scores:")
        print(raw_scores[0, head_index])
        
        print("\nScaled scores:")
        print(scaled_scores[0, head_index])

        print("\nMasked scores:")
        print(masked_scores[0, head_index])
        
        print("\nAttention weights:")
        print(attention_weights[0, head_index])

        print("\nAttention-weights row sums:")
        print(
            attention_weights[
                0,
                head_index,
            ].sum(dim=-1)
        )

        print("\nAttention output:")
        print(manual_output[0, head_index])
    print("\n--- Correctness Checks ---")
    print(
        "Future attention weights are zero:",
        torch.allclose(
            future_weights,
            torch.zeros_like(future_weights),
            atol=1e-7,
        ),
    )

    print(
        "Manual output matches PyTorch SDPA:",
        torch.allclose(
            manual_output,
            sdpa_output,
            atol=1e-6,
        ),
    )

    print("\n============================")
    print(" Attention experiment completed")
    print("=================================")


if __name__ == "__main__":
    main()
