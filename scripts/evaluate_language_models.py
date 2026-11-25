
import csv
import math
from pathlib import Path

from language_model.corpus import load_corpus
from language_model.preprocessing import tokenize
from language_model.vocabulary import (
    build_vocabulary,
    replace_unknown_tokens,
)
from language_model.ngrams import count_all_ngrams
from language_model.probability import build_context_counts
from language_model.evaluation import calculate_perplexity


TRAIN_PATH = "data/raw/wikitext-2/train.parquet"
VALIDATION_PATH = "data/raw/wikitext-2/validation.parquet"
TEST_PATH = "data/raw/wikitext-2/test.parquet"

OUTPUT_PATH = Path("reports/perplexity_results.csv")

MIN_FREQUENCY = 2


def prepare_documents(path):
    """Load and tokenize a corpus, preserving document boundaries."""

    documents = load_corpus(path)

    return [
        tokens
        for document in documents
        if document.strip()
        if (tokens := tokenize(document))
    ]


def format_perplexity(value):
    if math.isinf(value):
        return "infinity"

    return f"{value:,.4f}"


def evaluate_split(
    split_name,
    documents,
    all_counts,
    prediction_vocabulary_size,
):
    """Evaluate all three models using MLE and Laplace smoothing."""

    results = []

    print(f"\n{split_name.upper()} PERPLEXITY")
    print("-" * 70)
    print(f"{'Model':<15} {'MLE':>20} {'Laplace':>25}")
    print("-" * 70)

    for n in (1, 2, 3):

        counts = all_counts[n]

        contexts = (
            None
            if n == 1
            else build_context_counts(counts)
        )

        mle = calculate_perplexity(
            documents,
            n,
            counts,
            contexts,
        )

        laplace = calculate_perplexity(
            documents,
            n,
            counts,
            contexts,
            smoothing="laplace",
            vocabulary_size=prediction_vocabulary_size,
        )

        print(
            f"{n}-gram{'':<10}"
            f"{format_perplexity(mle):>20}"
            f"{format_perplexity(laplace):>25}"
        )

        results.append({
            "split": split_name,
            "model": f"{n}-gram",
            "mle_perplexity": format_perplexity(mle),
            "laplace_perplexity": format_perplexity(laplace),
        })

    print("-" * 70)

    return results


def main():

    print("LOADING DATA")
    print("-" * 50)

    train_documents = prepare_documents(TRAIN_PATH)
    validation_documents = prepare_documents(VALIDATION_PATH)
    test_documents = prepare_documents(TEST_PATH)

    print(f"Training documents:   {len(train_documents):,}")
    print(f"Validation documents: {len(validation_documents):,}")
    print(f"Test documents:       {len(test_documents):,}")

    # Learn vocabulary exclusively from training data.
    vocabulary = build_vocabulary(
        train_documents,
        min_frequency=MIN_FREQUENCY,
    )

    train_documents = replace_unknown_tokens(
        train_documents,
        vocabulary,
    )

    validation_documents = replace_unknown_tokens(
        validation_documents,
        vocabulary,
    )

    test_documents = replace_unknown_tokens(
        test_documents,
        vocabulary,
    )

    # Include </s> as a prediction token, but not <s>.
    prediction_vocabulary_size = len(vocabulary) + 1

    print(f"\nVocabulary size: {len(vocabulary):,}")
    print(f"Prediction vocabulary size: {prediction_vocabulary_size:,}")

    # Learn all model counts exclusively from training data.
    all_counts = count_all_ngrams(train_documents)

    print("\nTRAINING COMPLETE")

    results = []

    results.extend(
        evaluate_split(
            "validation",
            validation_documents,
            all_counts,
            prediction_vocabulary_size,
        )
    )

    results.extend(
        evaluate_split(
            "test",
            test_documents,
            all_counts,
            prediction_vocabulary_size,
        )
    )

    # Save the results.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "split",
                "model",
                "mle_perplexity",
                "laplace_perplexity",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nRESULTS SAVED TO: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
