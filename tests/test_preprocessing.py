from language_model.preprocessing import tokenize


def test_lowercase():
    assert tokenize("The CAT") == ["the", "cat"]


def test_punctuation():
    assert tokenize("Hello, world!") == ["hello", "world"]


def test_hyphen_marker():
    assert tokenize("real @-@ time") == ["real-time"]


def test_unicode():
    assert tokenize("Senjō café") == ["senjō", "café"]


def test_apostrophe():
    assert tokenize("don't stop") == ["don't", "stop"]


def test_empty_text():
    assert tokenize("") == []

def test_possessive_apostrophe():
    assert tokenize("the game 's story") == ["the", "game's", "story"]


def test_contraction_with_separated_apostrophe():
    assert tokenize("do n't stop") == ["don't", "stop"]
def test_plural_possessive():
    assert tokenize("girls' friendly society") == ["girls", "friendly", "society"]