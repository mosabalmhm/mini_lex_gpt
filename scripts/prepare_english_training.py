from pathlib import Path

from mini_lex_gpt.data.english_corpus import (
    save_english_corpus,
)
from mini_lex_gpt.tokenization.bpe_tokenizer import (
    train_bpe_tokenizer,
)

def main() -> None:

    processed_directory = Path(
        "data/processed"
    )

    tokenizer_path = Path(
        "data/tokenizer/tokenizer.json"
    )

    (
        dictionary_path,
        dialogue_path,
    ) = save_english_corpus(
        output_directory=processed_directory,
        max_dictionary_words=2000,
    )

    tokenizer = train_bpe_tokenizer(
        corpus_files=[
            dictionary_path,
            dialogue_path,
        ],
        output_path=tokenizer_path,
        vocab_size=4096,
    )
    
    print("===========================")
    print(" MiniLexGPT English Data")
    print("============================")

    print("\nDictionary:")
    print(dictionary_path)

    print("\nDialogues:")
    print(dialogue_path)

    print("\nTokenizer:")
    print(tokenizer_path)

    print("\nVocabulary size:")
    print(
        tokenizer.get_vocab_size()
    )

    print("\nSpecial tokens:")

    for token in [
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "<user>",
        "<assistant>",
        "<dict>",
    ]:
        print(
            token,
            tokenizer.token_to_id(
                token
            ),
        )

if __name__ == "__main__":
    main()

