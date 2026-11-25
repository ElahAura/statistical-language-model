
from language_model.corpus import load_corpus
from language_model.preprocessing import tokenize
from language_model.vocabulary import (
    build_vocabulary,
    replace_unknown_tokens,
)
from language_model.ngrams import count_all_ngrams
from language_model.probability import build_context_counts
from language_model.smoothing import laplace_probability


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

    unigram_counts = all_counts[1]
    bigram_counts = all_counts[2]
    trigram_counts = all_counts[3]

    # <s> is a context marker, not a prediction token.
    prediction_vocabulary_size = len(vocabulary) + 1

    bigram_contexts = build_context_counts(bigram_counts)
    trigram_contexts = build_context_counts(trigram_counts)

    examples = [
        (
            "Unigram",
            ("the",),
            unigram_counts,
            {},
        ),
        (
            "Bigram",
            ("of", "the"),
            bigram_counts,
            bigram_contexts,
        ),
        (
            "Trigram",
            ("one", "of", "the"),
            trigram_counts,
            trigram_contexts,
        ),
    ]

    print("\nLAPLACE SMOOTHING RESULTS")
    print("-" * 50)

    for model_name, ngram, counts, contexts in examples:
        probability = laplace_probability(
            ngram,
            counts,
            contexts,
            prediction_vocabulary_size,
        )

        print(f"\n{model_name}: {' '.join(ngram)}")
        print(f"Smoothed probability: {probability:.10f}")

    # Test an unseen bigram.
    unseen_bigram = ("of", "unseenword")

    unseen_probability = laplace_probability(
        unseen_bigram,
        bigram_counts,
        bigram_contexts,
        prediction_vocabulary_size,
    )

    print("\nUNSEEN N-GRAM TEST")
    print("-" * 50)
    print(f"Bigram: {' '.join(unseen_bigram)}")
    print(f"Smoothed probability: {unseen_probability:.10f}")
    print(f"Probability is nonzero: {unseen_probability > 0}")


if __name__ == "__main__":
    main()
