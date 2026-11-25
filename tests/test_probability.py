
import pytest
from collections import Counter

from language_model.probability import (
    ngram_probability,
    build_context_counts,
)


def test_unigram_probability():
    counts = Counter({
        ("the",): 2,
        ("cat",): 1,
        ("dog",): 1,
    })

    assert ngram_probability(("the",), counts) == 0.5


def test_bigram_probability():
    counts = Counter({
        ("of", "the"): 8,
        ("of", "a"): 1,
        ("of", "course"): 1,
    })

    assert ngram_probability(("of", "the"), counts) == 0.8


def test_trigram_probability():
    counts = Counter({
        ("one", "of", "the"): 9,
        ("one", "of", "these"): 1,
    })

    assert ngram_probability(("one", "of", "the"), counts) == 0.9


def test_probability_with_start_token():
    counts = Counter({
        ("<s>", "the"): 8,
        ("<s>", "a"): 2,
    })

    assert ngram_probability(("<s>", "the"), counts) == 0.8


def test_end_token_probability():
    counts = Counter({
        ("cat", "</s>"): 5,
        ("cat", "sat"): 5,
    })

    assert ngram_probability(("cat", "</s>"), counts) == 0.5


def test_unseen_ngram_probability():
    counts = Counter({
        ("of", "the"): 8,
        ("of", "a"): 2,
    })

    assert ngram_probability(("of", "course"), counts) == 0.0


def test_unseen_context_probability():
    counts = Counter({
        ("of", "the"): 8,
    })

    assert ngram_probability(("under", "the"), counts) == 0.0


def test_empty_ngram():
    with pytest.raises(TypeError):
        ngram_probability((), Counter())


def test_invalid_ngram_type():
    with pytest.raises(TypeError):
        ngram_probability(["of", "the"], Counter())


def test_invalid_counts_type():
    with pytest.raises(TypeError):
        ngram_probability(("of", "the"), {})


def test_build_context_counts():
    counts = Counter({
        ("of", "the"): 8,
        ("of", "a"): 2,
        ("in", "the"): 5,
    })

    contexts = build_context_counts(counts)

    assert contexts[("of",)] == 10
    assert contexts[("in",)] == 5
