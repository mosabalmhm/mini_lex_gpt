from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

def require_token_id(
    tokenizer: Tokenizer,
    token: str,
) -> int:

    token_id = tokenizer.token_to_id(
        token
    )

    if token_id is None:
        raise RuntimeError(
            f"Tokenizer is missing {token}."
        )

    return token_id

def build_training_token_stream(
    tokenizer: Tokenizer,
    dictionary_path: Path,
    dialogue_path: Path,
    seed: int = 32,
) -> torch.Tensor:

    bos_id = require_token_id(
        tokenizer,
        "<bos>",
    )

    eos_id = require_token_id(
        tokenizer,
        "<eos>",
    )

    user_id = require_token_id(
        tokenizer,
        "<user>",
    )

    assistant_id = require_token_id(
        tokenizer,
        "<assistant>",
    )

    dictionary_id = require_token_id(
        tokenizer,
        "<dict>",
    )

    records: list[list[int]] = []

    with dictionary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:

            text = line.strip()
            if not text:
                continue

            text_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
            ).ids

            record = [
                bos_id,
                dictionary_id,
                *text_ids,
                eos_id,
            ]

            records.append(
                record
            )

    with dialogue_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.reader(
            file,
            delimiter="\t",
        )

        for row in reader:

            if len(row) != 2:
                continue

            user_text = row[0]
            assistant_text = row[1]

            user_tokens = tokenizer.encode(
                user_text,
                add_special_tokens=False,
            ).ids

            assistant_tokens = (
                tokenizer.encode(
                    assistant_text,
                    add_special_tokens=False,
                ).ids
            )
            
            record = [
                bos_id,
                user_id,
                *user_tokens,
                assistant_id,
                *assistant_tokens,
                eos_id,
            ]

            records.append(
                record
            )

    random_generator = random.Random(
        seed
    )

    random_generator.shuffle(
        records
    )

    flat_token_ids = [
        token_id
        for record in records
        for token_id in record
    ]

    return torch.tensor(
        flat_token_ids,
        dtype=torch.long,
    )
