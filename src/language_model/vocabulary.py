
from collections import Counter


UNKNOWN_TOKEN = "<UNK>"


def build_vocabulary(
    tokenized_documents: list[list[str]],
    min_frequency: int = 2,
) -> dict[str, int]:
    """
    Build a vocabulary, mapping rare words to <UNK>.
    """

    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1.")

    if not all(
        isinstance(document, list)
        and all(isinstance(token, str) for token in document)
        for document in tokenized_documents
    ):
        raise TypeError("Expected a list of tokenized documents.")

    all_tokens = [
        token
        for document in tokenized_documents
        for token in document
    ]

    original_counts = Counter(all_tokens)

    vocabulary = Counter()

    for token, count in original_counts.items():
        if count >= min_frequency:
            vocabulary[token] = count
        else:
            vocabulary[UNKNOWN_TOKEN] += count

    return dict(vocabulary)


def replace_unknown_tokens(
    tokenized_documents: list[list[str]],
    vocabulary: dict[str, int],
) -> list[list[str]]:
    """
    Replace tokens absent from the vocabulary with <UNK>.
    """

    if UNKNOWN_TOKEN not in vocabulary:
        raise ValueError("Vocabulary must contain <UNK>.")

    known_words = set(vocabulary) - {UNKNOWN_TOKEN}

    return [
        [
            token if token in known_words else UNKNOWN_TOKEN
            for token in document
        ]
        for document in tokenized_documents
    ]


def vocabulary_size(vocabulary: dict[str, int]) -> int:
    return len(vocabulary)
