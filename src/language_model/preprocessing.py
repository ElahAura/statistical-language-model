
import re
import unicodedata


def tokenize(text: str) -> list[str]:
    """
    Normalize and tokenize text.

    Handles Unicode words, WikiText-2 hyphens,
    separated possessive apostrophes and contractions.
    """

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    text = text.lower()
    text = unicodedata.normalize("NFC", text)

    # Restore WikiText-2 hyphen markers.
    text = re.sub(r"\s*@-@\s*", "-", text)

    # Rejoin split negative contractions: "do n't" -> "don't".
    text = re.sub(r"\b(\w+)\s+(n['’]t)\b", r"\1\2", text)

    # Rejoin separated apostrophes: "game 's" -> "game's".
    text = re.sub(r"(?<=\w)\s+(['’])(?=\w)", r"\1", text)

    # Extract words, including internal hyphens and apostrophes.
    tokens = re.findall(
        r"[^\W\d_]+(?:[-'][^\W\d_]+)*",
        text,
        flags=re.UNICODE
    )

    return tokens
