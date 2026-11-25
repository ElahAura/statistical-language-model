
from pathlib import Path

from language_model.corpus import load_corpus
from language_model.preprocessing import tokenize
from language_model.vocabulary import (
    build_vocabulary,
    replace_unknown_tokens,
)
from language_model.ngrams import count_all_ngrams
from language_model.probability import build_context_counts


TRAIN_PATH = "data/raw/wikitext-2/train.parquet"

MIN_FREQUENCY = 2
TOLERANCE = 1e-9


def prepare_documents(path):
    documents = load_corpus(path)

    return [
        tokens
        for document in documents
        if document.strip()
        if (tokens := tokenize(document))
    ]



def audit_distribution(ngram_counts, context_counts, n):
    probability_sums = {}

    for ngram, count in ngram_counts.items():
        context = ngram[:-1]
        probability_sums[context] = (
            probability_sums.get(context, 0.0)
            + count / context_counts[context]
        )

    errors = [
        (context, probability_sum)
        for context, probability_sum in probability_sums.items()
        if abs(probability_sum - 1.0) > TOLERANCE
    ]

    return len(probability_sums), errors



def main():
    print("LOADING TRAINING CORPUS")
    print("-" * 55)

    documents = prepare_documents(TRAIN_PATH)

    vocabulary = build_vocabulary(
        documents,
        min_frequency=MIN_FREQUENCY,
    )

    documents = replace_unknown_tokens(
        documents,
        vocabulary,
    )

    all_counts = count_all_ngrams(documents)

    print(f"Documents:       {len(documents):,}")
    print(f"Vocabulary size: {len(vocabulary):,}")

    for n in (2, 3):

        print(f"\n{n}-GRAM MLE DISTRIBUTION AUDIT")
        print("-" * 55)

        counts = all_counts[n]
        contexts = build_context_counts(counts)

        checked, errors = audit_distribution(
            counts,
            contexts,
            n,
        )

        print(f"Contexts checked: {checked:,}")
        print(f"Invalid distributions: {len(errors):,}")

        if errors:
            print("\nFirst invalid distributions:")
            for context, probability_sum in errors[:10]:
                print(context, probability_sum)
        else:
            print("PASS: Every observed context sums to 1.")

    print("\nAUDIT COMPLETE")


if __name__ == "__main__":
    main()
