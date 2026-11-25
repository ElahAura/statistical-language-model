
from language_model.corpus import load_corpus
from language_model.preprocessing import tokenize
from language_model.vocabulary import (
    build_vocabulary,
    replace_unknown_tokens,
    vocabulary_size,
)


TRAIN_PATH = "data/raw/wikitext-2/train.parquet"
MIN_FREQUENCY = 2


def main():
    documents = load_corpus(TRAIN_PATH)

    tokenized_documents = [
        tokens
        for document in documents
        if document.strip()
        if (tokens := tokenize(document))
    ]

    original_vocabulary = build_vocabulary(
        tokenized_documents,
        min_frequency=1,
    )

    vocabulary = build_vocabulary(
        tokenized_documents,
        min_frequency=MIN_FREQUENCY,
    )

    processed_documents = replace_unknown_tokens(
        tokenized_documents,
        vocabulary,
    )

    original_token_count = sum(original_vocabulary.values())
    processed_token_count = sum(
        len(document) for document in processed_documents
    )

    print("UNKNOWN-WORD PROCESSING RESULTS")
    print("-" * 40)
    print("Minimum frequency:", MIN_FREQUENCY)
    print("Original vocabulary size:", vocabulary_size(original_vocabulary))
    print("New vocabulary size:", vocabulary_size(vocabulary))
    print("Original token count:", original_token_count)
    print("Processed token count:", processed_token_count)
    print("Tokens mapped to <UNK>:", vocabulary["<UNK>"])

    print("\nTwenty most frequent vocabulary entries:")
    for word, count in sorted(
        vocabulary.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:20]:
        print(f"{word:<20} {count}")


if __name__ == "__main__":
    main()
