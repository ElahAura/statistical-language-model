
from collections import Counter


def laplace_probability(
    ngram: tuple[str, ...],
    ngram_counts: Counter,
    context_counts: dict[tuple[str, ...], int],
    vocabulary_size: int,
) -> float:
    """
    Calculate the Add-One (Laplace) smoothed probability
    of an n-gram.
    """

    if not isinstance(ngram, tuple) or not ngram:
        raise TypeError("ngram must be a non-empty tuple.")

    if not all(isinstance(token, str) for token in ngram):
        raise TypeError("All n-gram elements must be strings.")

    if not isinstance(ngram_counts, Counter):
        raise TypeError("ngram_counts must be a Counter.")

    if not isinstance(context_counts, dict):
        raise TypeError("context_counts must be a dictionary.")

    if not isinstance(vocabulary_size, int) or vocabulary_size < 1:
        raise ValueError("vocabulary_size must be a positive integer.")

    numerator = ngram_counts[ngram] + 1

    if len(ngram) == 1:
        denominator = sum(ngram_counts.values()) + vocabulary_size
    else:
        history = ngram[:-1]
        denominator = context_counts.get(history, 0) + vocabulary_size

    return numerator / denominator
