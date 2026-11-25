from collections import Counter
from language_model.corpus import load_corpus

train_path = "data/raw/wikitext-2/train.parquet"

documents = load_corpus(train_path)

empty_entries = sum(not line.strip() for line in documents)

hyphen_markers = sum(line.count("@-@") for line in documents)

headings = [
    line for line in documents
    if line.strip().startswith("=")
]

print("Total entries:", len(documents))
print("Empty entries:", empty_entries)
print("Hyphen markers:", hyphen_markers)
print("Possible headings:", len(headings))

print("\nExamples of headings:")
for heading in headings[:5]:
    print(repr(heading))

print("\nMost common whitespace-separated tokens:")
tokens = Counter(
    token
    for line in documents
    for token in line.split()
)

print(tokens.most_common(20))