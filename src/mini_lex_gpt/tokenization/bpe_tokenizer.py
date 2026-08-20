from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import (
    ByteLevel as ByteLevelDecoder,
)

from tokenizers.models import (
    BPE,
)
from tokenizers.normalizers import (
    Lowercase,
    NFKC,
    Sequence,
)
from tokenizers.pre_tokenizers import (
    ByteLevel,
)
from tokenizers.trainers import (
    BpeTrainer,
)

SPICIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<user>",
    "<assistant>",
    "<dict>",
]

def train_bpe_tokenizer(
    corpus_files: list[Path],
    output_path: Path,
    vocab_size: int = 4096,
) -> Tokenizer:

    tokenizer = Tokenizer(
        BPE(
            unk_token="<unk>",
        )
    )

    tokenizer.normalizer = Sequence(
        [
            NFKC(),
            Lowercase(),
        ]
    )
    
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPICIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
    )

    tokenizer.train(
        files=[
            str(path)
            for path in corpus_files
        ],
        trainer=trainer,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(
        str(output_path)
    )

    for expected_id, token in enumerate(
        SPICIAL_TOKENS
    ):
        actual_id = tokenizer.token_to_id(
            token
        )

        if actual_id != expected_id:
            raise RuntimeError(
                f"Special token {token} "
                f"expected ID {expected_id}, "
                f"but received {actual_id}."
            )

    if tokenizer.get_vocab_size() != vocab_size:
        raise RuntimeError(
            "Tokenizer vocabulary size does not "
            "match model vocabulary size."
        )

    return tokenizer

def load_bpe_tokenizer(
    tokenizer_path: Path,
) -> Tokenizer:

    return Tokenizer.from_file(
        str(tokenizer_path)
    )
