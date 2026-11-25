
from language_model.corpus import load_corpus
from language_model.preprocessing import tokenize
from language_model.vocabulary import (
    build_vocabulary,
    replace_unknown_tokens,
)
from language_model.ngrams import count_all_ngrams
from language_model.probability import (
    ngram_probability,
    build_context_counts,
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

    # Maximum Likelihood Estimates
    examples = [
        ("Unigram", ("the",), unigram_counts),
        ("Bigram", ("of", "the"), bigram_counts),
        ("Trigram", ("one", "of", "the"), trigram_counts),
    ]

    print("\nMAXIMUM LIKELIHOOD ESTIMATES")
    print("-" * 50)

    for model_name, ngram, counts in examples:
        probability = ngram_probability(ngram, counts)

        print(f"\n{model_name}: {' '.join(ngram)}")
        print(f"Probability: {probability:.8f}")
        print(f"Percentage:  {probability * 100:.4f}%")

    # Probability Distribution Checks
    print("\nPROBABILITY DISTRIBUTION CHECKS")
    print("-" * 50)

    bigram_contexts = build_context_counts(bigram_counts)
    trigram_contexts = build_context_counts(trigram_counts)

    bigram_total = sum(
        bigram_counts[ngram] / bigram_contexts[ngram[:-1]]
        for ngram in bigram_counts
        if ngram[:-1] == ("of",)
    )

    trigram_total = sum(
        trigram_counts[ngram] / trigram_contexts[ngram[:-1]]
        for ngram in trigram_counts
        if ngram[:-1] == ("one", "of")
    )

    print(f"Bigram probabilities after 'of': {bigram_total:.8f}")
    print(f"Trigram probabilities after 'one of': {trigram_total:.8f}")

    print("\nVALIDATION")
    print("-" * 50)

    print(
        "Bigram distribution valid:",
        abs(bigram_total - 1.0) < 1e-9,
    )

    print(
        "Trigram distribution valid:",
        abs(trigram_total - 1.0) < 1e-9,
    )


if __name__ == "__main__":
    main()
