from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import nltk
from datasets import load_dataset
from nltk.corpus import wordnet as wn

WORD_PATTERN = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?"
)

def normalize_text(
    text: str,
) -> str:

    return " ".join(
        text.split()
    )

def load_short_english_dialogues(
    max_pairs: int,
) -> list[tuple[str, str]]:

    dataset = load_dataset(
        "OpenAssistant/oasst1",
        split="train",
    )

    messages_by_id = {
        row["message_id"]: row
        for row in dataset
    }

    pairs: list[
        tuple[str, str]
    ] = []

    assistant_messages = 0
    english_assistant_messages = 0
    valid_parent_messages = 0

    for row in dataset:
        
        role = str(
            row["role"]
        ).strip().lower()

        language = str(
            row["lang"]
        ).strip().lower()

        if role != "asssistant":
            continue
        assistant_messages += 1

        if language != "en":
            continue
        english_assistant_messages += 1

        if bool( row["deleted"]):
            continue
        
        parent_id = row["parent_id"]

        if parent_id is None:
            continue
 
        parent = messages_by_id.get(
            parent_id
        )

        if parent is None:
            continue

        parent_role = str(
            parent["role"]
        ).strip().lower()
            
        
        parent_language = str(
            parent["lang"]
        ).strip().lower()

        if parent_role != "prompter":
            continue

        if parent_language != "en":
            continue
        
        if bool(parent["deleted"]):
            continue

        valid_parent_messages += 1

        user_text = normalize_text(
            str(parent["text"])
        )

        assistant_text = normalize_text(
            str(row["text"])
        )

        user_word_count = len(
            user_text.split()
        )

        assistant_word_count = len(
            assistant_text.split()
        )

        if not (
            1 <= user_word_count <= 60
        ):
            continue

        if not (
            1 <= assistant_word_count <= 100
        ):
            continue

        if "```" in assistant_text:
            continue

        if "```" in user_text:
            continue

        if "```" in assistant_text:
            continue

        if "http://" in user_text:
            continue
 
        if "https://" in user_text:
            continue

        if "http://" in assistant_text:
            continue

        if "https://" in assistant_text:
            continue

        pairs.append(
            (
                user_text,
                assistant_text,
            )
        )

        if len(pairs) >= max_pairs:
            break

    print("\n--- OASST1 filtering ---")

    print(
        "Dataset rows:",
        len(dataset),
    )

    print(
        "Assistant messages:",
        assistant_messages,
    )

    print(
        "English assistant messages:",
        english_assistant_messages,
    )

    print(
        "English assistant/prompter pairs:",
        valid_parent_messages,
    )

    print(
        "Selected dialogue pairs:",
        len(pairs),
    )

    if len(pairs) == 0:
        raise RuntimeError(
            "No English dialogue pairs were "
            "selected from OASST1."
        )

    return pairs

def build_dictionary_entries(
    dialogue_pairs: list[
        tuple[str, str]
    ],
    max_words: int,
) -> list[str]:

    nltk.download(
        "wordnet",
        quiet=True,
    )

    word_counts: Counter[str] = Counter()

    for user_text, assistant_text in (
        dialogue_pairs
    ):

        combined_text = (
            user_text
            + " "
            + assistant_text
        )

        words = WORD_PATTERN.findall(
            combined_text.lower()
        )

        word_counts.update(
            words
        )

    dictionary_entries: list[str] = []

    for word, _ in (
        word_counts.most_common()
    ):

        synsets = wn.synsets(
            word
        )

        if len(synsets) == 0:
            continue

        synset = synsets[0]

        definition = normalize_text(
            synset.definition()
        )

        entry = (
            f"{word} means {definition}."
        )
     
        examples = synset.examples()

        if len(examples) > 0:
            example = normalize_text(
                examples[0]
            )

            entry += (
                f" example: {examle}"
            )

        dictionary_entries.append(
            entry
        )

        if (
            len(dictionary_entries)
            >= max_words
        ):
            break

    return dictionary_entries

def save_english_corpus(
    output_directory: Path,
    max_dialog_pairs: int = 4000,
    max_dictionary_words: int = 2000,
) -> tuple[Path, Path]:

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dialogue_pairs = (
        load_short_english_dialogues(
            max_pairs=max_dialog_pairs,
        )
    )

    dictionary_entries = (
        build_dictionary_entries(
            dialogue_pairs=dialogue_pairs,
            max_words=max_dictionary_words,
        )
    )

    dictionary_path = (
        output_directory
        / "dictionary.txt"
    )

    dialogue_path = (
        output_directory
        / "dialogues,tsv"
    )

    with dictionary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for entry in dictionary_entries:

            file.write(
                entry + "\n"
            )

    with dialogue_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.writer(
            file,
            delimiter="\t",
        )

        for user_text, assistant_text in (
            dialogue_pairs
        ):

            writer.writerow(
                [
                    user_text,
                    assistant_text,
                ]
            )

    return (
        dictionary_path,
        dialogue_path,
    )
