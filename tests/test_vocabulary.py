
import pytest

from language_model.vocabulary import (
    UNKNOWN_TOKEN,
    build_vocabulary,
    replace_unknown_tokens,
    vocabulary_size,
)


def test_build_vocabulary():
    documents = [
        ["the", "cat", "sat"],
        ["the", "dog", "sat"],
    ]

    vocabulary = build_vocabulary(documents)

    assert vocabulary == {
        "the": 2,
        "sat": 2,
        UNKNOWN_TOKEN: 2,
    }


def test_vocabulary_size():
    vocabulary = {
        "the": 2,
        "cat": 1,
        "dog": 1,
    }

    assert vocabulary_size(vocabulary) == 3


def test_empty_documents():
    assert build_vocabulary([]) == {}


def test_invalid_input():
    with pytest.raises(TypeError):
        build_vocabulary([["the", 123]])


def test_replace_unknown_tokens():
    documents = [["the", "cat", "giraffe"]]
    vocabulary = {"the": 5, "cat": 3, UNKNOWN_TOKEN: 2}

    result = replace_unknown_tokens(documents, vocabulary)

    assert result == [["the", "cat", UNKNOWN_TOKEN]]


def test_min_frequency():
    documents = [["cat", "dog", "cat", "bird"]]

    vocabulary = build_vocabulary(documents, min_frequency=2)

    assert vocabulary == {
        "cat": 2,
        UNKNOWN_TOKEN: 2,
    }


def test_invalid_min_frequency():
    with pytest.raises(ValueError):
        build_vocabulary([["cat"]], min_frequency=0)


def test_missing_unknown_token():
    with pytest.raises(ValueError):
        replace_unknown_tokens(
            [["giraffe"]],
            {"cat": 2},
        )
