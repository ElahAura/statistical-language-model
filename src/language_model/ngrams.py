
from collections import Counter


START_TOKEN = "<s>"
END_TOKEN = "</s>"


def count_ngrams(
    tokenized_documents: list[list[str]],
    n: int,
) -> Counter:
    """
    Count n-grams across independent documents.

    Adds beginning and ending markers appropriate
    to the n-gram order.
    """

    if n < 1:
        raise ValueError("n must be at least 1.")

    if not all(
        isinstance(document, list)
        and all(isinstance(token, str) for token in document)
        for document in tokenized_documents
    ):
        raise TypeError("Expected a list of tokenized documents.")

    ngram_counts = Counter()

    for document in tokenized_documents:

        sequence = (
            [START_TOKEN] * (n - 1)
            + document
            + [END_TOKEN]
        )

        for i in range(len(sequence) - n + 1):
            ngram = tuple(sequence[i:i + n])
            ngram_counts[ngram] += 1

    return ngram_counts


def count_all_ngrams(
    tokenized_documents: list[list[str]],
) -> dict[int, Counter]:
    """
    Count unigrams, bigrams, and trigrams.
    """

    return {
        n: count_ngrams(tokenized_documents, n)
        for n in (1, 2, 3)
    }
