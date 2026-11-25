
import math

from language_model.ngrams import START_TOKEN, END_TOKEN


def calculate_perplexity(
    documents: list[list[str]],
    n: int,
    ngram_counts,
    context_counts: dict[tuple[str, ...], int] | None = None,
    smoothing: str | None = None,
    vocabulary_size: int | None = None,
) -> float:
    """
    Calculate corpus-level perplexity for an n-gram model.

    smoothing:
        None       -> Maximum Likelihood Estimation
        "laplace"  -> Add-One smoothing
    """

    if n < 1:
        raise ValueError("n must be at least 1.")

    if smoothing not in (None, "laplace"):
        raise ValueError("smoothing must be None or 'laplace'.")

    if n > 1 and context_counts is None:
        raise ValueError("context_counts are required for n > 1.")

    if smoothing == "laplace":
        if (
            not isinstance(vocabulary_size, int)
            or isinstance(vocabulary_size, bool)
            or vocabulary_size < 1
        ):
            raise ValueError(
                "A positive vocabulary_size is required for Laplace smoothing."
            )

    # Precompute denominators once, rather than once per token.
    if n == 1:
        unigram_total = sum(ngram_counts.values())

        if smoothing == "laplace":
            unigram_denominator = unigram_total + vocabulary_size
        else:
            unigram_denominator = unigram_total

    total_log_probability = 0.0
    total_predictions = 0
    zero_probability_found = False

    for document in documents:

        if n == 1:
            sequence = document + [END_TOKEN]
        else:
            sequence = (
                [START_TOKEN] * (n - 1)
                + document
                + [END_TOKEN]
            )

        for i in range(n - 1, len(sequence)):

            ngram = tuple(sequence[i - n + 1:i + 1])

            numerator = ngram_counts[ngram]

            if smoothing == "laplace":
                numerator += 1

                if n == 1:
                    denominator = unigram_denominator
                else:
                    denominator = (
                        context_counts.get(ngram[:-1], 0)
                        + vocabulary_size
                    )

            elif n == 1:
                denominator = unigram_denominator

            else:
                denominator = context_counts.get(ngram[:-1], 0)

            probability = (
                numerator / denominator
                if denominator > 0
                else 0.0
            )

            total_predictions += 1

            if probability == 0:
                zero_probability_found = True
            else:
                total_log_probability += math.log(probability)

    if total_predictions == 0:
        raise ValueError("Cannot calculate perplexity on an empty corpus.")

    if zero_probability_found:
        return math.inf

    return math.exp(-total_log_probability / total_predictions)
