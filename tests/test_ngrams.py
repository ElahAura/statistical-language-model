
import pytest

from language_model.ngrams import (
    count_ngrams,
    count_all_ngrams,
)


def test_unigram_counts():
    documents = [["the", "cat", "the"]]

    counts = count_ngrams(documents, 1)

    assert counts[("the",)] == 2
    assert counts[("cat",)] == 1
    assert counts[("</s>",)] == 1


def test_bigram_counts():
    documents = [["the", "cat"]]

    counts = count_ngrams(documents, 2)

    assert counts[("<s>", "the")] == 1
    assert counts[("the", "cat")] == 1
    assert counts[("cat", "</s>")] == 1


def test_trigram_counts():
    documents = [["the", "cat"]]

    counts = count_ngrams(documents, 3)

    assert counts[("<s>", "<s>", "the")] == 1
    assert counts[("<s>", "the", "cat")] == 1
    assert counts[("the", "cat", "</s>")] == 1


def test_documents_are_independent():
    documents = [["the", "cat"], ["a", "dog"]]

    counts = count_ngrams(documents, 2)

    assert counts[("cat", "a")] == 0
    assert counts[("the", "cat")] == 1
    assert counts[("a", "dog")] == 1


def test_invalid_n():
    with pytest.raises(ValueError):
        count_ngrams([["the"]], 0)


def test_count_all_ngrams():
    documents = [["the", "cat"]]

    counts = count_all_ngrams(documents)

    assert set(counts.keys()) == {1, 2, 3}