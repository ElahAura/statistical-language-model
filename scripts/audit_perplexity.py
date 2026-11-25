
import math

from language_model.ngrams import count_all_ngrams
from language_model.probability import build_context_counts
from language_model.evaluation import calculate_perplexity


# Tiny controlled training corpus.
training_documents = [
    ["a", "b"],
    ["a", "c"],
]

# Evaluation sequence.
evaluation_documents = [
    ["a", "b"],
]

# Prediction targets: a, b, c, and </s>.
vocabulary_size = 4

# Build n-gram counts.
all_counts = count_all_ngrams(training_documents)

print("PERPLEXITY IMPLEMENTATION AUDIT")
print("=" * 55)

expected_results = {
    1: {
        "MLE": 54 ** (1 / 3),
        "Laplace": (500 / 9) ** (1 / 3),
    },
    2: {
        "MLE": 2 ** (1 / 3),
        "Laplace": 15 ** (1 / 3),
    },
    3: {
        "MLE": 2 ** (1 / 3),
        "Laplace": 15 ** (1 / 3),
    },
}

all_passed = True

for n in (1, 2, 3):

    counts = all_counts[n]

    contexts = (
        None if n == 1
        else build_context_counts(counts)
    )

    print(f"\n{n}-GRAM MODEL")
    print("-" * 55)

    for smoothing in (None, "laplace"):

        model_name = "MLE" if smoothing is None else "Laplace"

        actual = calculate_perplexity(
            documents=evaluation_documents,
            n=n,
            ngram_counts=counts,
            context_counts=contexts,
            smoothing=smoothing,
            vocabulary_size=(
                vocabulary_size if smoothing == "laplace" else None
            ),
        )

        expected = expected_results[n][model_name]

        passed = math.isclose(
            actual,
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

        all_passed = all_passed and passed

        print(f"{model_name:10} Actual:   {actual:.12f}")
        print(f"{'':10} Expected: {expected:.12f}")
        print(f"{'':10} Result:   {'PASS' if passed else 'FAIL'}")

print("\n" + "=" * 55)

if all_passed:
    print("AUDIT PASSED: All perplexity calculations are correct.")
else:
    print("AUDIT FAILED: At least one calculation differs.")

print("=" * 55)
