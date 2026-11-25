
import math

import pytest

from language_model.evaluation import calculate_perplexity
from language_model.ngrams import count_ngrams
from language_model.probability import build_context_counts


def test_bigram_perplexity_on_observed_sequence():
    documents = [["a"]]
    counts = count_ngrams(documents, 2)
    contexts = build_context_counts(counts)

    perplexity = calculate_perplexity(
        documents,
        2,
        counts,
        contexts,
    )

    assert perplexity == pytest.approx(1.0)


def test_trigram_perplexity_on_observed_sequence():
    documents = [["a", "b"]]
    counts = count_ngrams(documents, 3)
    contexts = build_context_counts(counts)

    perplexity = calculate_perplexity(
        documents,
        3,
        counts,
        contexts,
    )

    assert perplexity == pytest.approx(1.0)


def test_unigram_perplexity():
    documents = [["a"]]
    counts = count_ngrams(documents, 1)

    perplexity = calculate_perplexity(
        documents,
        1,
        counts,
    )

    assert perplexity == pytest.approx(2.0)


def test_mle_returns_infinity_for_unseen_ngram():
    training_documents = [["a", "b"]]
    test_documents = [["a", "c"]]

    counts = count_ngrams(training_documents, 2)
    contexts = build_context_counts(counts)

    perplexity = calculate_perplexity(
        test_documents,
        2,
        counts,
        contexts,
    )

    assert math.isinf(perplexity)


def test_laplace_handles_unseen_ngram():
    training_documents = [["a", "b"]]
    test_documents = [["a", "c"]]

    counts = count_ngrams(training_documents, 2)
    contexts = build_context_counts(counts)

    perplexity = calculate_perplexity(
        test_documents,
        2,
        counts,
        contexts,
        smoothing="laplace",
        vocabulary_size=4,
    )

    assert math.isfinite(perplexity)
    assert perplexity > 0


def test_empty_corpus_raises_error():
    with pytest.raises(ValueError):
        calculate_perplexity([], 2, count_ngrams([], 2), {})
