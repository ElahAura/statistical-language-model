
import pytest
from collections import Counter

from language_model.smoothing import laplace_probability


def test_unigram_laplace_probability():
    counts = Counter({
        ("the",): 2,
        ("cat",): 1,
    })

    probability = laplace_probability(
        ("the",), counts, {}, 3
    )

    assert probability == pytest.approx(3 / 6)


def test_bigram_laplace_probability():
    counts = Counter({
        ("of", "the"): 8,
        ("of", "a"): 2,
    })

    contexts = {
        ("of",): 10,
    }

    probability = laplace_probability(
        ("of", "the"), counts, contexts, 5
    )

    assert probability == pytest.approx(9 / 15)


def test_trigram_laplace_probability():
    counts = Counter({
        ("one", "of", "the"): 9,
        ("one", "of", "these"): 1,
    })

    contexts = {
        ("one", "of"): 10,
    }

    probability = laplace_probability(
        ("one", "of", "the"), counts, contexts, 5
    )

    assert probability == pytest.approx(10 / 15)


def test_unseen_ngram_has_nonzero_probability():
    counts = Counter({
        ("of", "the"): 8,
        ("of", "a"): 2,
    })

    contexts = {
        ("of",): 10,
    }

    probability = laplace_probability(
        ("of", "course"), counts, contexts, 5
    )

    assert probability > 0


def test_unseen_context_has_nonzero_probability():
    counts = Counter({
        ("of", "the"): 8,
    })

    probability = laplace_probability(
        ("under", "the"), counts, {}, 5
    )

    assert probability == pytest.approx(1 / 5)


def test_invalid_vocabulary_size():
    with pytest.raises(ValueError):
        laplace_probability(
            ("of", "the"), Counter(), {}, 0
        )
