
from collections import Counter, defaultdict


def ngram_probability(
    ngram: tuple[str, ...],
    ngram_counts: Counter,
) -> float:
    """
    Calculate the Maximum Likelihood Estimate (MLE)
    for an n-gram.
    """

    if not isinstance(ngram, tuple) or not ngram:
        raise TypeError("ngram must be a non-empty tuple.")

    if not all(isinstance(token, str) for token in ngram):
        raise TypeError("All n-gram elements must be strings.")

    if not isinstance(ngram_counts, Counter):
        raise TypeError("ngram_counts must be a Counter.")

    if not ngram_counts:
        return 0.0

    numerator = ngram_counts[ngram]

    if len(ngram) == 1:
        denominator = sum(ngram_counts.values())
    else:
        history = ngram[:-1]

        denominator = sum(
            count
            for candidate, count in ngram_counts.items()
            if candidate[:-1] == history
        )

    if denominator == 0:
        return 0.0

    return numerator / denominator


def build_context_counts(
    ngram_counts: Counter,
) -> dict[tuple[str, ...], int]:
    """
    Precompute the total frequency of every context.

    This avoids repeatedly scanning all n-grams.
    """

    context_counts = defaultdict(int)

    for ngram, count in ngram_counts.items():
        if len(ngram) > 1:
            context_counts[ngram[:-1]] += count

    return dict(context_counts)
