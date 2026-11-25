
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from language_model.corpus import load_corpus
from language_model.preprocessing import tokenize


TRAIN_PATH = "data/raw/wikitext-2/train.parquet"

CSV_PATH = Path("reports/vocabulary_growth.csv")
FIGURE_PATH = Path("reports/figures/vocabulary_growth.png")

RANDOM_SEED = 42

CHECKPOINTS = [
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.40,
    0.60,
    0.80,
    1.00,
]


def load_tokenized_documents(path):
    documents = load_corpus(path)

    return [
        tokens
        for document in documents
        if document.strip()
        if (tokens := tokenize(document))
    ]


def measure_vocabulary_growth(documents):
    # Shuffle documents reproducibly.
    documents = documents.copy()
    random.Random(RANDOM_SEED).shuffle(documents)

    all_tokens = [
        token
        for document in documents
        for token in document
    ]

    total_tokens = len(all_tokens)

    checkpoints = [
        max(1, int(total_tokens * proportion))
        for proportion in CHECKPOINTS
    ]

    vocabulary = set()
    results = []
    checkpoint_index = 0

    for index, token in enumerate(all_tokens, start=1):
        vocabulary.add(token)

        while (
            checkpoint_index < len(checkpoints)
            and index >= checkpoints[checkpoint_index]
        ):
            results.append({
                "tokens_processed": index,
                "vocabulary_size": len(vocabulary),
                "percentage_processed": (
                    index / total_tokens * 100
                ),
            })

            checkpoint_index += 1

    return results


def estimate_heaps_exponent(results):
    token_counts = np.array([
        row["tokens_processed"]
        for row in results
    ], dtype=float)

    vocabulary_sizes = np.array([
        row["vocabulary_size"]
        for row in results
    ], dtype=float)

    log_tokens = np.log(token_counts)
    log_vocabulary = np.log(vocabulary_sizes)

    slope, intercept = np.polyfit(
        log_tokens,
        log_vocabulary,
        1,
    )

    predicted = intercept + slope * log_tokens

    ss_res = np.sum((log_vocabulary - predicted) ** 2)
    ss_tot = np.sum(
        (log_vocabulary - np.mean(log_vocabulary)) ** 2
    )

    r_squared = 1 - ss_res / ss_tot

    return slope, np.exp(intercept), r_squared


def save_results(results):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "tokens_processed",
                "vocabulary_size",
                "percentage_processed",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nCSV SAVED TO: {CSV_PATH}")


def create_plot(results, beta, constant):
    token_counts = np.array([
        row["tokens_processed"]
        for row in results
    ], dtype=float)

    vocabulary_sizes = np.array([
        row["vocabulary_size"]
        for row in results
    ], dtype=float)

    fitted = constant * token_counts ** beta

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.loglog(
        token_counts,
        vocabulary_sizes,
        marker="o",
        linewidth=2,
        label="Observed vocabulary growth",
    )

    ax.loglog(
        token_counts,
        fitted,
        linestyle="--",
        label=f"Heaps' law fit (β = {beta:.3f})",
    )

    ax.set_title("Vocabulary Growth in WikiText-2")
    ax.set_xlabel("Tokens processed (log scale)")
    ax.set_ylabel("Distinct word types (log scale)")

    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    fig.tight_layout()

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"FIGURE SAVED TO: {FIGURE_PATH}")


def main():
    print("ANALYZING VOCABULARY GROWTH")
    print("-" * 55)

    documents = load_tokenized_documents(TRAIN_PATH)

    results = measure_vocabulary_growth(documents)

    beta, constant, r_squared = estimate_heaps_exponent(
        results
    )

    print(f"Total documents: {len(documents):,}")
    print(
        f"Total tokens:    "
        f"{results[-1]['tokens_processed']:,}"
    )

    print("\nVOCABULARY GROWTH")
    print("-" * 55)
    print(
        f"{'Tokens processed':>20} "
        f"{'Vocabulary size':>20} "
        f"{'Corpus processed':>20}"
    )

    for row in results:
        print(
            f"{row['tokens_processed']:>20,} "
            f"{row['vocabulary_size']:>20,} "
            f"{row['percentage_processed']:>19.0f}%"
        )

    print("\nHEAPS' LAW ESTIMATION")
    print("-" * 55)
    print(f"Estimated beta: {beta:.4f}")
    print(f"Estimated K:    {constant:.4f}")
    print(f"R-squared:      {r_squared:.4f}")

    save_results(results)
    create_plot(results, beta, constant)


if __name__ == "__main__":
    main()
