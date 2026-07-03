# VecFuzz: Vector-based Fuzzy Matching

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A **fast approximate string matching library** that turns words into compact vectors so you can find the closest match, even when the query is riddled with typos! It’s a fuzzy matching on steroids: sub‑millisecond lookup times and linear memory scaling.

## Why VecFuzz?

Fuzzy matching techniques usually suffer from a three-way trade-off: speed, accuracy, and memory. Many approaches are excel at one, but degrade sharply on another, especially as typo tolerance increases.

VecFuzz is designed to improve them together rather than trading one off against another, placing it on the Pareto frontier:

- **Speed:** FAISS‑powered HNSW index gives you ~14 000 queries per second **on a laptop CPU** (and even faster with a GPU!).
- **Accuracy:** It correctly catches ~99 % of transpositions errors, beating all other algorithms by a wide margin.
- **Memory:** The index size grows **linearly** with dictionary size. No exponential blow‑up from edit distance.

If you’ve ever wished for a fuzzy matching distance algorithm that runs at hash‑table speed, VecFuzz is for you!

---

## Features

- **Deterministic vectorisation:** Same word always gives the same vector.
- **FAISS‑backed:** Choose between CPU or GPU indexing for massive throughput.
- **Linear memory:** Index size is `O(dictionary_size * vector_size)`, regardless of error tolerance.
- **Multilingual by design:** Built to stay language-neutral and work across different writing systems without built-in bias.
- **Simple API:** Build, save, load, and query with a few lines of Python.
- **Typo‑tolerant, not a typo‑corrector:** Use it anywhere you need approximate string matching: record linkage, OCR post‑processing, fuzzy search, or duplicate detection.

---

## How It Works

VecFuzz converts each word into a fixed‑length vector that encodes four complementary views of the string. A frequency and position vector to handle small local perturbations (insertions and transpositions), and a preceding and succeeding vector to handle character-identity loss (substitutions and deletions).

1. **Character frequency:** How often each letter appears.
2. **Average position:** Where each letter tends to sit in the word.
3. **Preceding characters:** Weighted influence of letters that come before.
4. **Succeeding characters:** Weighted influence of letters that come after.

All vectors are normalized by word length, so "apple" and "apples" end up close together.

After vectorizing the dictionary, we build a **FAISS HNSW index** using Manhattan (L1) distance. A query vector is searched in this index, and the nearest neighbours become your top‑k candidates. The whole process is deterministic, interpretable, and very, very fast.

---

## Benchmark Highlights

### Synthetic Dataset
Here’s a comparison on a dictionary of ~160 000 English words (we kept only words with 4+ characters), tested with 5 000 randomly generated misspellings (25% of substitutions, insertions, deletions, and transposition). 80% of misspelled words contains 1 error, 20% contains 2 errors. All measurements are averaged over 5 trials. Tested on a Ryzen 9 365.

#### Overall Accuracy and Speed

| Method | Top-1 | Top-3 | Top-5 | Duration (s) | Build (s) | Size (MB) |
|---|---|---|---|---|---|---|
| VecFuzz              | 84.05% 🥇   | 93.39% 🥇   | 95.61% 🥇   | 0.382 🥈   | 24.090  | 111.37    |
| SymSpell             | 78.53%     | 90.76%     | 92.94%     | 0.170 🥇   | 1.982  | 190.17    |
| RapidFuzz            | 80.10% 🥈   | 91.85%     | 94.72% 🥉   | 55.423    | 0.0     | 0.0      |
| Jaro-Winkler         | 79.68% 🥉   | 92.33% 🥈   | 94.76% 🥈   | 71.860    | 0.0    | 0.0      |
| Damerau-Levenshtein  | 79.06%     | 91.95% 🥉   | 94.47%     | 528.623   | 0.0     | 0.0      |
| Levenshtein          | 69.53%     | 85.02%     | 88.98%     | 62.813    | 0.0     | 0.0      |
| Norvig               | 78.40%     | 89.90%     | 92.10%     | 44.230 🥉  | 0.0     | 0.0      |

#### Top‑1 Accuracy by Error Type

| Method | Substitution | Insertion | Deletion | Transposition |
|----------------------|--------------|-----------|----------|---------------|
| VecFuzz              |   72.6%     |   96.3% 🥈  |   67.9% 🥉 |   98.9% 🥇    |
| SymSpell             |   80.8% 🥉    |   94.9%   |   53.9%  |   84.4%     |
| RapidFuzz            |   71.3%     |   97.8% 🥇  |   79.8% 🥇 |   71.0%     |
| Jaro-Winkler         |   60.2%     |   95.3% 🥉  |   73.5% 🥈 |   88.9% 🥈    |
| Damerau-Levenshtein  |   81.7% 🥈    |   94.5%   |   55.3%  |   84.7% 🥉    |
| Levenshtein          |   81.8% 🥇    |   94.5%   |   55.6%  |   46.4%     |
| Norvig               |   79.9%     |   95.1%   |   54.6%  |   83.9%     |

#### Top‑1 Accuracy by Error Count and Error Position

| Method | 1‑Error | 2‑Errors | Prefix | Middle | Suffix |
|----------------------|---------|----------|--------|--------|--------|
| VecFuzz              |   88.3% 🥇 |    67.3% 🥇 |   80.8% 🥇 |   89.5% 🥇|   75.4% 🥇 |
| SymSpell             |   82.7% |    61.7% |         74.4% 🥉|   85.9% |   66.5% |
| RapidFuzz            |   83.8% 🥈|    65.5% 🥉 |   73.8% |   86.7% 🥉|   72.1% 🥈 |
| Jaro-Winkler         |   83.1% 🥉 |    65.9% 🥈 |   71.8% |   87.1% 🥈|   71.6% 🥉 |
| Damerau-Levenshtein  |   83.0% |        62.8% |   76.0% 🥈|   86.3% |   66.3% |
| Levenshtein          |   73.3% |        54.6% |   63.9% |   76.5% |   60.1% |
| Norvig               |   82.6% |        61.5% |   74.1% |   85.9% |   66.4% |


### Real-World Human Error Benchmark (Birkbeck Spelling Error Corpus)
Here’s a comparison on the same dictionary of ~160 000 English words (all of it), tested with the Birkbeck Spelling Error Corpus. This dataset consists of non-synthetic human misspellings including heavy phonetic mutations, dysgraphia and multi-error handwriting slips. Tested on a Ryzen 9 365.

| Method | Top-1 | Top-5 | Top-10 | Top-25 | Top-100 | Duration (s) | Build (s) | Size (MB) |
|--------|-----------|-----------|------------|------------|-------------|--------------|-----------|-----------|
| VecFuzz              | 31.77%     | 49.62% 🥉  | 56.16% 🥉  | 64.05% 🥉 | 73.15%      | 3.413 🥇      | 24.107   | 112.51    |
| SymSpell             | 34.06% 🥇   | 48.92%    | 51.94%     | 54.58%     | 57.70%      | 12.596 🥈     | 37.902   | 3568.23   |
| RapidFuzz            | 32.65% 🥉   | 51.74% 🥇   | 58.54% 🥇   | 66.56% 🥇   | 76.67% 🥇    | 412.887 🥉    | 0.0       | 0.0       |
| Jaro-Winkler         | 30.27%     | 50.72% 🥈   | 57.66% 🥈   | 65.43% 🥈   | 75.86% 🥈    | 518.326      | 0.0       | 0.0       |
| Damerau-Levenshtein  | 29.20%     | 48.10%     | 55.56%   | 63.92%      | 73.18% 🥉    | 3431.185     | 0.0       | 0.0       |
| Levenshtein          | 28.10%     | 46.73%     | 54.20%     | 62.64%     | 72.35%      | 463.465      | 0.0       | 0.0       |
| Norvig               | 33.77% 🥈   | 40.80%     | 41.33%     | 41.48%     | 41.48%      | 746.374      | 0.0       | 0.0       |

---

## Installation

No pip package yet, the project is under active development. Import it directly as `from vecfuzz import VecFuzz` after placing `vecfuzz.py` and `faiss_index.py` in your working directory.
Clone or download this repository, make sure you have Python 3.8 or newer, and install the required dependencies.

### Dependencies:

- `faiss-cpu` (or `faiss-gpu` if you have an NVIDIA GPU and CUDA)
- `numpy`

Install everything with:

```bash
pip install faiss-cpu numpy
```

---

## Quick Start

The repository includes examples files that you can run immediately under `/examples`. Note that `vecfuzz.py` must be in the same directory. Below is a compact walk‑through.

```python
from vecfuzz import VecFuzz

# Build a small list of words
words = ["apple", "banana", "orange", "peach", "pineapple"]

# Create the index and build it
index = VecFuzz().index(words)

# Look up 3 nearest neighbours for each fuzzy query
queries = ["aple", "bannana", "orng"]
results = index.lookup(queries, k=3)

for query, candidates in results:
    print(f"Candidates for '{query}':")
    for candidate, distance in candidates:
        print(f"  → {candidate} (L1 distance: {distance:.4f})")

# Expected output:
#    Candidates for 'aple':
#      → apple (L1 distance: 2.7016)
#      → pineapple (L1 distance: 10.7789)
#      → peach (L1 distance: 12.5534)
#    Candidates for 'bannana':
#      → banana (L1 distance: 2.5323)
#      → orange (L1 distance: 18.1574)
#      → peach (L1 distance: 20.7256)
#    Candidates for 'orng':
#      → orange (L1 distance: 6.9839)
#      → banana (L1 distance: 17.3495)
#      → apple (L1 distance: 19.3234)
```

---

## Contributing

This is an early‑stage tool, and I’d love your ideas. Open an issue or pull request if you find a bug, have a feature request, or want to improve performance.

Please keep the code clean, the documentation human‑friendly, and the benchmarks honest.

---

## License

VecFuzz is provided under the MIT License. Use it, modify it, ship it in your product. Just retain the copyright notice.

If you build something interesting with it, I’d love to hear from you!
