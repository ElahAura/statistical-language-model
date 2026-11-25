
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from language_model.corpus import load_corpus
from language_model.preprocessing import tokenize
from language_model.vocabulary import (
    build_vocabulary,
    replace_unknown_tokens,
)
from language_model.ngrams import count_all_ngrams, END_TOKEN


TRAIN_PATH = "data/raw/wikitext-2/train.parquet"
VALIDATION_PATH = "data/raw/wikitext-2/validation.parquet"
TEST_PATH = "data/raw/wikitext-2/test.parquet"

MIN_FREQUENCY = 2

CSV_PATH = Path("reports/error_analysis.csv")
FIGURE_PATH = Path("reports/figures/unseen_ngrams.png")


def prepare_documents(path):
    documents = load_corpus(path)

    return [
        tokens
        for document in documents
        if document.strip()
        if (tokens := tokenize(document))
    ]


def get_prediction_ngrams(documents, n):
    counts = Counter()

    for document in documents:
        if n == 1:
            sequence = document + [END_TOKEN]
        else:
            sequence = ["<s>"] * (n - 1) + document + [END_TOKEN]

        for i in range(n - 1, len(sequence)):
            ngram = tuple(sequence[i - n + 1:i + 1])
            counts[ngram] += 1

    return counts


def analyze_split(split_name, original_documents, vocabulary,
                  training_counts):

    documents = replace_unknown_tokens(original_documents, vocabulary)

    known_words = set(vocabulary) - {"<UNK>"}
    total_tokens = sum(len(doc) for doc in original_documents)

    oov_tokens = sum(
        1
        for doc in original_documents
        for token in doc
        if token not in known_words
    )

    oov_rate = oov_tokens / total_tokens * 100

    print(f"\n{split_name.upper()} ERROR ANALYSIS")
    print("=" * 60)
    print(f"OOV tokens: {oov_tokens:,} / {total_tokens:,}")
    print(f"OOV rate:   {oov_rate:.4f}%")

    results = []

    for n in (1, 2, 3):
        heldout_ngrams = get_prediction_ngrams(documents, n)
        train_ngrams = training_counts[n]

        unseen = Counter({
            ngram: frequency
            for ngram, frequency in heldout_ngrams.items()
            if ngram not in train_ngrams
        })

        total_events = sum(heldout_ngrams.values())
        unseen_occurrences = sum(unseen.values())
        unseen_rate = unseen_occurrences / total_events * 100

        unseen_context_types = 0
        events_with_unseen_context = 0

        if n > 1:
            train_contexts = {
                ngram[:-1] for ngram in train_ngrams
            }

            heldout_contexts = {
                ngram[:-1] for ngram in heldout_ngrams
            }

            unseen_contexts = heldout_contexts - train_contexts
            unseen_context_types = len(unseen_contexts)

            events_with_unseen_context = sum(
                frequency
                for ngram, frequency in heldout_ngrams.items()
                if ngram[:-1] not in train_contexts
            )

        print(f"\n{n}-gram")
        print(f"  Total events:             {total_events:,}")
        print(f"  Unseen types:             {len(unseen):,}")
        print(f"  Unseen occurrences:       {unseen_occurrences:,}")
        print(f"  Unseen occurrence rate:   {unseen_rate:.4f}%")
        print(f"  Unseen context types:     {unseen_context_types:,}")
        print(f"  Events with unseen context: {events_with_unseen_context:,}")

        print("  Most frequent unseen n-grams:")
        for ngram, frequency in unseen.most_common(5):
            print(f"    {' '.join(ngram):<30} {frequency:>8,}")

        results.append({
            "split": split_name,
            "ngram_order": n,
            "total_events": total_events,
            "oov_tokens": oov_tokens,
            "total_word_tokens": total_tokens,
            "oov_rate_percent": round(oov_rate, 4),
            "unseen_types": len(unseen),
            "unseen_occurrences": unseen_occurrences,
            "unseen_rate_percent": round(unseen_rate, 4),
            "unseen_context_types": unseen_context_types,
            "events_with_unseen_context": events_with_unseen_context,
        })

    return results


def save_csv(results):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCSV SAVED TO: {CSV_PATH}")


def create_figure(results):
    splits = ["validation", "test"]
    orders = [1, 2, 3]

    values = {
        split: [
            next(
                row["unseen_rate_percent"]
                for row in results
                if row["split"] == split
                and row["ngram_order"] == n
            )
            for n in orders
        ]
        for split in splits
    }

    positions = range(len(orders))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        [p - width / 2 for p in positions],
        values["validation"],
        width,
        label="Validation",
    )

    ax.bar(
        [p + width / 2 for p in positions],
        values["test"],
        width,
        label="Test",
    )

    ax.set_title("Unseen N-gram Occurrence Rate")
    ax.set_xlabel("N-gram order")
    ax.set_ylabel("Unseen prediction events (%)")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(["Unigram", "Bigram", "Trigram"])
    ax.set_ylim(0, 80)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"FIGURE SAVED TO: {FIGURE_PATH}")


def main():
    print("LOADING CORPUS")
    print("-" * 50)

    train_documents = prepare_documents(TRAIN_PATH)
    validation_documents = prepare_documents(VALIDATION_PATH)
    test_documents = prepare_documents(TEST_PATH)

    vocabulary = build_vocabulary(
        train_documents,
        min_frequency=MIN_FREQUENCY,
    )

    processed_train = replace_unknown_tokens(
        train_documents,
        vocabulary,
    )

    training_counts = count_all_ngrams(processed_train)

    print(f"Training documents: {len(train_documents):,}")
    print(f"Vocabulary size:    {len(vocabulary):,}")

    results = []

    results.extend(
        analyze_split(
            "validation",
            validation_documents,
            vocabulary,
            training_counts,
        )
    )

    results.extend(
        analyze_split(
            "test",
            test_documents,
            vocabulary,
            training_counts,
        )
    )

    save_csv(results)
    create_figure(results)


if __name__ == "__main__":
    main()
