
from collections import Counter
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from language_model.corpus import load_corpus
from language_model.preprocessing import tokenize


TRAIN_PATH = "data/raw/wikitext-2/train.parquet"

CSV_PATH = Path("reports/word_frequencies.csv")
FIGURE_PATH = Path("reports/figures/zipf_law.png")


def load_word_frequencies(path):
    documents = load_corpus(path)

    frequencies = Counter()
    total_tokens = 0

    for document in documents:
        if not document.strip():
            continue

        tokens = tokenize(document)
        frequencies.update(tokens)
        total_tokens += len(tokens)

    return frequencies, total_tokens


def calculate_zipf_exponent(ranks, frequencies):
    """
    Estimate the Zipf exponent using log-log linear regression.

    Fit: log(frequency) = intercept + slope * log(rank)
    Zipf exponent = -slope
    """

    # Exclude the first few ranks to reduce the influence
    # of exceptionally frequent function words.
    selected = (ranks >= 10) & (ranks <= 10000)

    log_ranks = np.log(ranks[selected])
    log_frequencies = np.log(frequencies[selected])

    slope, intercept = np.polyfit(
        log_ranks,
        log_frequencies,
        1,
    )

    predicted = intercept + slope * log_ranks

    ss_res = np.sum((log_frequencies - predicted) ** 2)
    ss_tot = np.sum(
        (log_frequencies - np.mean(log_frequencies)) ** 2
    )

    r_squared = 1 - ss_res / ss_tot

    return -slope, r_squared


def save_frequency_table(words, counts, total_tokens):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "rank",
            "word",
            "frequency",
            "relative_frequency",
        ])

        for rank, (word, count) in enumerate(
            zip(words, counts),
            start=1,
        ):
            writer.writerow([
                rank,
                word,
                count,
                count / total_tokens,
            ])

    print(f"\nCSV SAVED TO: {CSV_PATH}")


def create_zipf_plot(ranks, frequencies):
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.loglog(
        ranks,
        frequencies,
        linewidth=1.5,
        label="Observed word frequencies",
    )

    ax.set_title("Word Frequency Distribution (Zipf's Law)")
    ax.set_xlabel("Word rank (log scale)")
    ax.set_ylabel("Word frequency (log scale)")

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
    print("ANALYZING WORD FREQUENCIES")
    print("-" * 55)

    frequencies, total_tokens = load_word_frequencies(
        TRAIN_PATH
    )

    sorted_words = frequencies.most_common()

    words = [word for word, _ in sorted_words]
    counts = np.array(
        [count for _, count in sorted_words],
        dtype=np.float64,
    )

    ranks = np.arange(1, len(words) + 1)

    exponent, r_squared = calculate_zipf_exponent(
        ranks,
        counts,
    )

    print(f"Total tokens:       {total_tokens:,}")
    print(f"Unique word types:  {len(words):,}")

    print("\nTOP 20 MOST FREQUENT WORDS")
    print("-" * 55)

    for rank, (word, count) in enumerate(
        sorted_words[:20],
        start=1,
    ):
        percentage = count / total_tokens * 100

        print(
            f"{rank:>3}. {word:<20} "
            f"{count:>10,}  ({percentage:.3f}%)"
        )

    print("\nZIPF'S LAW ESTIMATION")
    print("-" * 55)
    print(f"Estimated exponent: {exponent:.4f}")
    print(f"R-squared:          {r_squared:.4f}")
    print("Fitted rank range:  10 to 10,000")

    save_frequency_table(
        words,
        counts,
        total_tokens,
    )

    create_zipf_plot(ranks, counts)


if __name__ == "__main__":
    main()
