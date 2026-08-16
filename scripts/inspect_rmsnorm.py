from __future__ import annotations

import torch

from mini_lex_gpt.model.rmsnorm import RMSNorm

def calculate_rms(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """

    Calculate RMS across the final dimension.
    """

    return torch.sqrt(
        tensor.pow(2).mean(dim=-1)
    )

def main() -> None:
    dimension=4
    epsilon=1e-5

    rms_norm = RMSNorm(
        dimension=dimension,
        epsilon=epsilon,
    )

    input_tensor = torch.tensor(
        [
            [3.0, 4.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [-2.0, 0.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    output_tensor = rms_norm(input_tensor)

    input_rms = calculate_rms(input_tensor)
    output_rms = calculate_rms(output_tensor)

    input_means = input_tensor.mean(dim=-1)
    output_means = output_tensor.mean(dim=-1)

    manual_mean_square = input_tensor.pow(2).mean(
        dim=-1,
        keepdim=True,
    )

    manual_output = input_tensor * torch.rsqrt(
        manual_mean_square + epsilon
    )

    manual_output = (
        manual_output * rms_norm.weight
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in rms_norm.parameters()
        if parameter.requires_grad
    )

    print("=========================")
    print(" MiniLexGPT RMSNorm")
    print("=========================")
 
    print("\n--- Layer ---")
    print(rms_norm)

    print("\n--- Input ---")
    print(input_tensor)

    print("\nInput shape:")
    print(input_tensor.shape)


    print("\nInput RMS values:")
    print(input_rms)

    print("\nInput means:")
    print(input_means)

    print("\n--- Output ---")
    print(output_tensor)

    print("\nOutput shape:")
    print(output_tensor.shape)

    print("\nOutput RMS values:")
    print(output_rms)

    print("\nOutput means:")
    print(output_means)

    print("\n--- Manual Formule Check ---")
    
    outputs_match = torch.allclose(
        output_tensor,
        manual_output,
        atol=1e-6,
    )

    print(
        "Layer output matches manual formula:",
        outputs_match,
    )

    rms_values_close_to_one = torch.allclose(
        output_rms,
        torch.ones_like(output_rms),
        atol=1e-5,
    )

    print(
        "Output RMS values are close to one:",
        rms_values_close_to_one,
    )

    print("\n--- Parameters ---")

    print("Weight:")
    print(rms_norm.weight)

    print("\nWeight shape:")
    print(rms_norm.weight.shape)

    print("\nTrainable parameter count:")
    print(parameter_count)

    print("\n--- Gradient Experiment ---")

    artificial_loss = output_tensor.sum()

    print("Artificial loss:")
    print(artificial_loss.item())

    artificial_loss.backward()

    print("\nWeight gradients:")
    print(rms_norm.weight.grad)

    print("\nInput gradients:")
    print(input_tensor.grad)

    print("\n===============================")
    print(" RMSNorm inspection completed ")
    print("==================================")

if __name__ == "__main__":
    main()
