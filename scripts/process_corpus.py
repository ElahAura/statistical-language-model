
from collections import Counter

from language_model.corpus import load_corpus
from language_model.preprocessing import tokenize


TRAIN_PATH = "data/raw/wikitext-2/train.parquet"


def main():
    documents = load_corpus(TRAIN_PATH)

    tokenized_documents = [
        tokens
        for document in documents
        if document.strip()
        if (tokens := tokenize(document))
    ]

    all_tokens = [
        token
        for sequence in tokenized_documents
        for token in sequence
    ]

    word_counts = Counter(all_tokens)

    print("CORPUS PROCESSING RESULTS")
    print("-" * 35)
    print("Original entries:", len(documents))
    print("Non-empty token sequences:", len(tokenized_documents))
    print("Total tokens:", len(all_tokens))
    print("Vocabulary size:", len(word_counts))

    print("\nTwenty most frequent words:")
    for word, count in word_counts.most_common(20):
        print(f"{word:<20} {count}")


if __name__ == "__main__":
    main()
