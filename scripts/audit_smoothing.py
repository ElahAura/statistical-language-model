
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



def audit_laplace(ngram_counts, context_counts, n, vocab_size):

    observed_probability_sums = {}
    observed_type_counts = {}

    if n == 1:
        denominator = sum(ngram_counts.values()) + vocab_size

        observed_probability_sum = sum(
            (count + 1) / denominator
            for count in ngram_counts.values()
        )

        unseen_types = vocab_size - len(ngram_counts)

        probability_sum = (
            observed_probability_sum
            + unseen_types / denominator
        )

        deviation = abs(probability_sum - 1.0)
        errors = int(deviation > TOLERANCE)

        return 1, errors, probability_sum

    for ngram, count in ngram_counts.items():
        context = ngram[:-1]

        denominator = context_counts[context] + vocab_size

        observed_probability_sums[context] = (
            observed_probability_sums.get(context, 0.0)
            + (count + 1) / denominator
        )

        observed_type_counts[context] = (
            observed_type_counts.get(context, 0) + 1
        )

    errors = 0
    maximum_deviation = 0.0

    for context, observed_sum in observed_probability_sums.items():

        denominator = context_counts[context] + vocab_size

        unseen_types = (
            vocab_size - observed_type_counts[context]
        )

        probability_sum = (
            observed_sum + unseen_types / denominator
        )

        deviation = abs(probability_sum - 1.0)
        maximum_deviation = max(maximum_deviation, deviation)

        if deviation > TOLERANCE:
            errors += 1

    return len(observed_probability_sums), errors, maximum_deviation



def main():
    print("LOADING TRAINING CORPUS")
    print("-" * 55)

    documents = prepare_documents(TRAIN_PATH)

    vocabulary = build_vocabulary(
        documents,
        min_frequency=MIN_FREQUENCY,
    )

    documents = replace_unknown_tokens(documents, vocabulary)

    all_counts = count_all_ngrams(documents)

    # Prediction vocabulary includes all vocabulary tokens and EOS.
    # BOS is a context marker, not a prediction target.
    prediction_vocab_size = len(vocabulary) + 1

    print(f"Vocabulary size:            {len(vocabulary):,}")
    print(f"Prediction vocabulary size:  {prediction_vocab_size:,}")

    for n in (1, 2, 3):
        print(f"\n{n}-GRAM LAPLACE AUDIT")
        print("-" * 55)

        counts = all_counts[n]

        contexts = (
            None if n == 1
            else build_context_counts(counts)
        )

        checked, errors, deviation = audit_laplace(
            counts,
            contexts,
            n,
            prediction_vocab_size,
        )

        print(f"Distributions checked: {checked:,}")
        print(f"Invalid distributions: {errors:,}")

        if n == 1:
            print(f"Probability sum:       {deviation:.12f}")
        else:
            print(f"Maximum deviation:     {deviation:.12e}")

        if errors == 0:
            print("PASS: All checked distributions sum to 1.")
        else:
            print("FAIL: Some distributions do not sum to 1.")

    print("\nAUDIT COMPLETE")


if __name__ == "__main__":
    main()
