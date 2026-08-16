from mini_lex_gpt.config import ModelConfig


def format_integer(value: int) -> str:
    """
    Format an integer using commas.
    """

    return f"{value:,}"


def main() -> None:
    """
    Create a configuration and inspect its architecture,
    parameter count, and estimated memory usage.
    """

    config = ModelConfig()
    breakdown = config.parameter_breakdown()

    print("====================================")
    print(" MiniLexGPT Model Configuration")
    print("====================================")

    print("\n--- Architecture ---")

    for name, value in config.to_dictionary().items():
        print(f"{name:22s}: {value}")

    print(
        f"{'head_dimension':22s}: "
        f"{config.head_dimension}"
    )

    print("\n--- Parameter Breakdown ---")

    fields_to_show = [
        "token_embedding",
        "attention_per_block",
        "swiglu_per_block",
        "rmsnorm_per_block",
        "one_transformer_block",
        "all_transformer_blocks",
        "final_rmsnorm",
        "language_model_head",
        "total",
    ]

    for field_name in fields_to_show:
        print(
            f"{field_name:25s}: "
            f"{format_integer(breakdown[field_name])}"
        )

    print("\n--- Memory Estimate ---")

    print(
        "Parameters only, float32:",
        f"{config.parameter_memory_mib():.2f} MiB"
    )

    print(
        "Parameters + gradients + AdamW states:",
        f"{config.adamw_training_memory_mib():.2f} MiB"
    )

    print(
        "\nImportant: activation memory, batches, "
        "and PyTorch overhead are not included."
    )

    print("\n====================================")
    print(" Configuration inspection completed")
    print("====================================")


if __name__ == "__main__":
    main()
