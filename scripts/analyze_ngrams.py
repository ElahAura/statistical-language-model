
from language_model.corpus import load_corpus
from language_model.preprocessing import tokenize
from language_model.vocabulary import (
    build_vocabulary,
    replace_unknown_tokens,
)
from language_model.ngrams import count_all_ngrams


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

    vocabulary = build_vocabulary(
        tokenized_documents,
        min_frequency=MIN_FREQUENCY,
    )

    processed_documents = replace_unknown_tokens(
        tokenized_documents,
        vocabulary,
    )

    all_counts = count_all_ngrams(processed_documents)

    for n, counts in all_counts.items():
        print(f"\nTOP 20 {n}-GRAMS")
        print("-" * 40)

        for ngram, frequency in counts.most_common(20):
            print(f"{' '.join(ngram):<30} {frequency}")


if __name__ == "__main__":
    main()
