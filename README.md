# Statistical Language Modelling Laboratory

A from-scratch implementation and experimental study of unigram, bigram, and trigram statistical language models using Python. The project investigates probability estimation, Laplace smoothing, perplexity, vocabulary growth, word-frequency distributions, and the effects of data sparsity on language modelling.

## 1. Project Overview

Statistical language models estimate the probability of a sequence of words. They are fundamental to natural language processing and provide a mathematical foundation for understanding how computational systems model language.

This project implements three classical n-gram language models:

- **Unigram:** estimates word probabilities independently.
- **Bigram:** estimates each word's probability given the preceding word.
- **Trigram:** estimates each word's probability given the preceding two words.

The models are implemented from scratch rather than using an existing language-modelling library.

The project focuses on understanding both the mathematical principles behind statistical language modelling and the practical challenges involved in building, evaluating, and auditing these models.

## 2. Dataset

The experiments use the **WikiText-2** dataset.

The dataset is divided into three splits:

- Training set
- Validation set
- Test set

The original dataset contains 36,718 training entries. After preprocessing, 23,694 non-empty token sequences remain.

Each non-empty dataset entry is treated as an independent sequence. Sequences are not concatenated, and document boundaries are preserved.

### Training corpus statistics

| Metric | Value |
|---|---:|
| Original training entries | 36,718 |
| Non-empty token sequences | 23,694 |
| Total tokens | 1,661,340 |
| Original vocabulary size | 72,294 |
| Vocabulary size after frequency filtering | 39,989 |

## 3. Text Preprocessing and Vocabulary

The preprocessing pipeline includes:

- Lowercasing
- Unicode normalization
- Restoration of WikiText-2 hyphen markers
- Handling of separated apostrophes and negative contractions
- Word tokenization

The vocabulary is constructed exclusively from the training data.

Words occurring fewer than two times are mapped to a special unknown token, `<UNK>`. Unseen words in the validation and test sets are also mapped to `<UNK>`.

This reduces the vocabulary while preserving the total number of token occurrences.

Special sequence-boundary tokens are used:

- `<s>`: beginning of a sequence
- `</s>`: end of a sequence
- `<UNK>`: unknown or infrequent word

Beginning-of-sequence tokens provide context for bigram and trigram models. Every sequence receives an end-of-sequence token.

## 4. Model Implementation

### 4.1 N-gram counting

The implementation extracts and counts unigram, bigram, and trigram occurrences.

For example, given:

`the cat sleeps`

The corresponding bigrams include:

`(<s>, the)`, `(the, cat)`, `(cat, sleeps)`, `(sleeps, </s>)`

The trigram model uses two beginning-of-sequence markers to provide sufficient initial context.

### 4.2 Maximum Likelihood Estimation

Maximum Likelihood Estimation (MLE) estimates probabilities from observed frequencies.

For a unigram:

\[
P(w)=\frac{C(w)}{\sum_{w'}C(w')}
\]

For a conditional n-gram:

\[
P(w_i\mid h)=\frac{C(h,w_i)}{C(h)}
\]

where \(h\) represents the preceding context.

MLE assigns zero probability to an n-gram that never occurs in the training corpus. This becomes a significant problem when evaluating unseen sequences.

### 4.3 Laplace Smoothing

To address zero probabilities, the project implements add-one (Laplace) smoothing.

\[
P_{\text{Laplace}}(w_i\mid h)
=
\frac{C(h,w_i)+1}{C(h)+V}
\]

where \(V\) is the prediction vocabulary size.

The prediction vocabulary contains 39,990 entries, including `<UNK>` and `</s>`, but excluding `<s>`.

Laplace smoothing assigns non-zero probability to unseen n-grams. However, add-one smoothing can distribute too much probability mass to unseen events, particularly in higher-order models.

## 5. Model Evaluation

The models are evaluated using **perplexity**, a standard measure of language-model performance.

\[
PP(W)=
\exp\left(
-\frac{1}{N}
\sum_{i=1}^{N}\log P(w_i\mid h_i)
\right)
\]

Lower perplexity indicates that a model assigns higher probability to the evaluated sequence, although comparisons must account for differences in model setup and evaluation data.

Evaluation includes all prediction events, including end-of-sequence tokens.

### Perplexity results

**Validation set**

| Model | MLE | Laplace |
|---|---:|---:|
| Unigram | 1,148.0200 | 1,154.8116 |
| Bigram | Infinity | 3,891.6000 |
| Trigram | Infinity | 20,526.7658 |

**Test set**

| Model | MLE | Laplace |
|---|---:|---:|
| Unigram | 1,108.8657 | 1,114.5293 |
| Bigram | Infinity | 3,725.9710 |
| Trigram | Infinity | 20,038.0091 |

### Interpretation

Several observations emerge:

1. Unigram MLE performs slightly better than unigram Laplace smoothing. Smoothing does not necessarily improve performance in every setting.
2. Bigram and trigram MLE produce infinite perplexity because at least one evaluation event receives zero probability.
3. Laplace smoothing eliminates zero probabilities and produces finite perplexity.
4. The trigram model performs substantially worse than the bigram model under Laplace smoothing. This is consistent with the increased sparsity of higher-order n-gram contexts.
5. The test-set perplexities are lower than the validation-set values. This indicates that the two splits differ in their difficulty under these models; it does not establish that the models generalize better to the test set.

## 6. Error and Data Sparsity Analysis

The project investigates why MLE models fail on held-out data and how vocabulary differences contribute to the problem.

### Out-of-vocabulary rates

| Dataset | OOV tokens | OOV rate |
|---|---:|---:|
| Validation | 9,345 | 5.3749% |
| Test | 12,367 | 6.3273% |

### Unseen n-gram occurrences

| Dataset | Model | Unseen occurrences | Percentage |
|---|---|---:|---:|
| Validation | Bigram | 47,900 | 27.1667% |
| Validation | Trigram | 120,579 | 68.3868% |
| Test | Bigram | 53,093 | 26.7681% |
| Test | Trigram | 134,530 | 67.8266% |

The analysis also distinguishes between unseen n-grams and unseen contexts.

For the bigram model, all evaluated one-word contexts were observed in training, but many word continuations were unseen.

For the trigram model, some two-word contexts were themselves absent from training. This creates an additional source of sparsity.

These findings help explain why higher-order MLE models produce zero probabilities and why simple add-one smoothing struggles as n-gram order increases.

## 7. Corpus Statistical Analysis

Beyond model construction and evaluation, the project investigates two well-known statistical properties of natural language.

### 7.1 Zipf's Law

Word frequencies are examined in relation to their rank in the corpus.

A log-log regression over ranks 10–10,000 produces:

| Metric | Result |
|---|---:|
| Estimated exponent | 1.0414 |
| R² | 0.9873 |

The results show a strong Zipf-like relationship between word frequency and rank over the selected range.

### 7.2 Heaps' Law

Vocabulary growth is examined as additional training tokens are observed.

The estimated relationship is:

\[
V(N)=KN^\beta
\]

where \(V(N)\) is the number of distinct word types observed after processing \(N\) tokens.

| Metric | Result |
|---|---:|
| Estimated \(K\) | 21.2926 |
| Estimated \(\beta\) | 0.5718 |
| R² | 0.9965 |

The observed vocabulary grows sublinearly with corpus size, consistent with Heaps-like vocabulary growth.

The estimates are empirical fits based on nine checkpoints and should not be interpreted as universal constants.

## 8. Implementation Validation and Testing

Correctness is treated as an important part of the project.

The implementation includes unit tests and independent numerical audits.

### Unit tests

The test suite covers:

- Text preprocessing
- Vocabulary construction
- N-gram counting
- Probability estimation
- Laplace smoothing
- Perplexity calculation

**Current status: 46 tests passing.**

### Probability normalization audit

The Laplace-smoothed probability distributions are checked to ensure that their probabilities sum to one.

| Model | Distributions checked | Invalid distributions |
|---|---:|---:|
| Unigram | 1 | 0 |
| Bigram | 39,990 | 0 |
| Trigram | 622,185 | 0 |

All checked distributions pass the normalization audit. The maximum deviations for bigram and trigram distributions are attributable to floating-point precision.

### Independent perplexity audit

A separate audit uses a small manually constructed corpus to compare the implementation against independently calculated expected perplexity values.

All six checks pass:

- Unigram MLE
- Unigram Laplace
- Bigram MLE
- Bigram Laplace
- Trigram MLE
- Trigram Laplace

This provides an additional check beyond the unit tests and full-corpus experiments.

## 9. Repository Structure

```text
statistical-language-model/
│
├── data/
│   ├── raw/
│   │   └── wikitext-2/
│   └── processed/
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   ├── error_analysis.csv
│   ├── perplexity_results.csv
│   ├── word_frequencies.csv
│   └── vocabulary_growth.csv
│
├── scripts/
│   ├── inspect_corpus.py
│   ├── process_corpus.py
│   ├── build_vocabulary.py
│   ├── analyze_ngrams.py
│   ├── analyze_probabilities.py
│   ├── analyze_smoothing.py
│   ├── evaluate_language_models.py
│   ├── analyze_errors.py
│   ├── analyze_frequency.py
│   ├── analyze_vocabulary_growth.py
│   ├── audit_probabilities.py
│   ├── audit_smoothing.py
│   └── audit_perplexity.py
│
├── src/
│   └── language_model/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── vocabulary.py
│       ├── corpus.py
│       ├── ngrams.py
│       ├── probability.py
│       ├── smoothing.py
│       └── evaluation.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_vocabulary.py
│   ├── test_ngrams.py
│   ├── test_probability.py
│   ├── test_smoothing.py
│   └── test_evaluation.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 10. Reproducing the Experiments

### Requirements

- Python
- Dependencies listed in `requirements.txt`
- WikiText-2 dataset files in Parquet format

The expected dataset files are:

```text
data/raw/wikitext-2/
├── train.parquet
├── validation.parquet
└── test.parquet
```

### Installation

Clone the repository and enter its directory:

```bash
git clone <repository-url>
cd statistical-language-model
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Running the tests

```bash
PYTHONPATH=src pytest -q
```

### Running the experiments

The scripts can be executed individually:

```bash
PYTHONPATH=src python scripts/inspect_corpus.py
PYTHONPATH=src python scripts/process_corpus.py
PYTHONPATH=src python scripts/build_vocabulary.py
PYTHONPATH=src python scripts/analyze_ngrams.py
PYTHONPATH=src python scripts/analyze_probabilities.py
PYTHONPATH=src python scripts/analyze_smoothing.py
PYTHONPATH=src python scripts/evaluate_language_models.py
PYTHONPATH=src python scripts/analyze_errors.py
PYTHONPATH=src python scripts/analyze_frequency.py
PYTHONPATH=src python scripts/analyze_vocabulary_growth.py
```

The audit scripts can also be run independently:

```bash
PYTHONPATH=src python scripts/audit_probabilities.py
PYTHONPATH=src python scripts/audit_smoothing.py
PYTHONPATH=src python scripts/audit_perplexity.py
```

Generated reports and figures are saved under `reports/`.

## 11. Limitations and Future Work

This project focuses on classical count-based language modelling. Its results should be understood in that context.

Current limitations include:

- Add-one smoothing is a simple baseline and is not generally the most effective smoothing technique.
- Higher-order n-gram models suffer from substantial data sparsity.
- The tokenizer is designed for the WikiText-2 corpus and does not attempt to solve every possible tokenization problem.
- The project does not yet implement neural language models or compare its results against pretrained models.

Potential future extensions include:

- Interpolated and backoff language models
- More advanced smoothing techniques, such as Kneser-Ney smoothing
- Improved tokenization and unknown-word handling
- Comparisons with neural language models
- Further analysis of language-model behaviour and generalization

## 12. References

- Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016). *Pointer Sentinel Mixture Models*. The WikiText language-modelling dataset was introduced in this work.
- Jurafsky, D., & Martin, J. H. *Speech and Language Processing*. Reference text for statistical language modelling, n-grams, and language-model evaluation.
