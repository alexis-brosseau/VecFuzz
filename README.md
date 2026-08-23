# VecFuzz: Vector-based Fuzzy Matching

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A **fast string matching library** that turns words into compact vectors so you can find the closest match, even when the query is riddled with typos! It delivers high recall on human misspellings while keeping memory usage flat and having sub-millisecond lookup speeds.


## The trade-off

Fuzzy matching has a three-way tension between speed, memory, and accuracy. VecFuzz is designed to be competitive on all three:

*   **Memory ($O(N)$ vs $O(N \cdot L^d)$)**: SymSpell’s index size is explodes combinatorially based on the max word length ($L$) and max edit distance ($d$). VecFuzz scales strictly linearly $O(N)$ with your dictionary size, completely independent of typo tolerance. Memory stays flat no matter how fuzzy you need to get.
*   **Build Speed ($O(N \log N)$ vs $O(N \cdot L^d)$)**: SymSpell has to generate and hash every possible delete variant, leading to massive build times at higher edit distances. VecFuzz vectorizes the corpus and builds a FAISS HNSW graph, operating in $O(N \log N)$ time. It’s a slower build than SymSpell's lowest config, but it doesn't exponentially punish you for higher accuracy.
*   **Lookup Speed ($O(\log N)$ vs $O(1)$)**: SymSpell achieves $O(1)$ hash lookups while VecFuzz traverses an HNSW graph in $O(\log N)$ time, but because it uses FAISS under the hood, it is **highly parallelizable**.

If your dictionary is large, your memory budget is tight, queries need to be fast, and you can tolerate a slower build, VecFuzz is a strong pick.


## How it works

Instead of a rigid, hardcoded pipeline, VecFuzz is built on a **composable vectorizer architecture**. 

When you initialize VecFuzz, you can pass a custom list of mathematical feature extractors via the `vectorizers` parameter. By default, it uses a highly optimized combination, but you can mix and match them to tune the model for your specific error types.

You can make your own vectorizer using the `@vectorizer` decorator or use some of the premade ones :
1.  **Character Frequency**: How often each letter appears.
2.  **Positional Density**: The sum of normalized positions before/after a character, capturing the "weight" of the word's structure.
3.  **Average Position**: The mean position of each character. 
4.  **Phase-encoded Position**: A sinusoidal (cos/sin) expansion across frequency bands to capture global positional awareness.
5.  **Radial Basis Function (RBF)**: A smooth, local Gaussian positional encoding.
6.  **Adjacency Bigrams**: A fixed-size hashed count of character pairs to capture local ordering.

All sub-vectors are normalized by word length, so "apple" and "apples" land close together. The sub-vectors are concatenated and indexed with FAISS HNSW under Manhattan (L1) distance; a query is vectorized the same way and the nearest neighbours in the index become the top-k candidates.

## Benchmark Highlights

### Real-world human errors (Birkbeck Spelling Error Corpus)
Tested on a ~160k word dictionary using non-synthetic human misspellings (phonetic errors, dysgraphia, multi-error handwriting slips) on a Ryzen 9 365.

#### 1. Recall Accuracy
VecFuzz achieves the best recall at every `k` threshold, comfortably beating brute-force methods and SymSpell.

| Method | Recall@1 | Recall@5 | Recall@10 | Recall@25 | Recall@100 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VecFuzz** | **36.33%** | **54.96%** | **61.39%** | **68.83%** | **77.94%** |
| SymSpell | 34.05% | 48.92% | 51.94% | 54.58% | 57.70% |
| RapidFuzz | 32.64% | 51.74% | 58.54% | 66.56% | 76.67% |
| Levenshtein | 28.10% | 46.73% | 54.20% | 62.64% | 72.35% |

#### 2. Performance & Footprint
This is where the Big O trade-offs become obvious. VecFuzz achieves similar lookup speed at a fraction of the memory of higher-order SymSpell configs.

| Method | Lookup (s) | Build (s) | Size (MB) |
| :--- | :--- | :--- | :--- |
| VecFuzz (16 threads) | **4.81** | 42.17 | 221.11 |
| VecFuzz (4 threads) | 16.60 | 182.25 | 221.11 |
| VecFuzz (1 thread) | 25.71 | 345.78 | 221.11 |
| SymSpell d4/p12 | 12.25 | 38.82 | 3568.23 |
| SymSpell d3/p9 | 7.88 | 8.79 | 842.84 |
| SymSpell d2/p7 | 7.38 | **2.03** | **190.88** |
| RapidFuzz | 403.63 | N/A | N/A |
| Levenshtein | 454.25 | N/A | N/A |

### Synthetic edit-distance sweep

Dictionary of 150k words, compared against SymSpell at three delete-distance/prefix-length configs (d2/p7, d3/p9, d4/p12).

*   **Substitutions**: SymSpell d4/p12 (**31.3%**) leads here over VecFuzz (**24.5%**), which remains the primary area for future algorithmic improvement.
*   **Transpositions**: VecFuzz (**87.6%**) starts near-perfect and stays well above SymSpell d4/p12 (**46.3%**).
*   **Insertions**: VecFuzz (**78.1%**) degrades gracefully even at 9 edits. SymSpell drops to **0%** once edits exceed its configured max distance.
*   **Deletions**: VecFuzz (**14.3%**) is ahead of SymSpell d4/p12 (**10.7%**), though both methods struggle heavily past 2-3 deletion edits.

![Accuracy by Error Type Chart](benchmark_outputs/accuracy_by_error_type_and_edits.png)

## When to use this
*   **Large dictionaries** where SymSpell-style precomputed edit indexes get too large to fit in memory.
*   **Interactive/high-QPS fuzzy search** where brute-force comparison (RapidFuzz, raw Levenshtein) is too slow.
*   **Multi-core environments** where you can leverage FAISS's multi-threading.

## When not to use this
*   **Small dictionaries** where memory and speed aren't a constraint.
*   **Workloads with frequent dictionary updates** (updating the HNSW graph takes longer than updating a SymSpell hash map).

## Installation

No pip package yet, the project is under active development. Import it directly as `from vecfuzz import VecFuzz` after placing `vecfuzz.py` in your working directory.
Clone or download this repository, make sure you have Python 3.8 or newer, and install the required dependencies.

### Dependencies:

- `faiss-cpu`
- `numpy`

Install everything with:

```bash
pip install -r requirements.txt
```

## Quick Start

The repository includes examples files that you can run immediately under `/examples`. Note that `vecfuzz.py` must be in the same directory. Below is a compact walk-through.

```python
from vecfuzz import VecFuzz

words = ["apple", "banana", "orange", "peach", "pineapple"]

vecfuzz = VecFuzz().build(words)

queries = ["aple", "bannana", "orng"]
results = vecfuzz.lookup(queries, k=3)

for query, candidates in results:
    print(f"Candidates for '{query}':")
    for candidate, distance in candidates:
        print(f"  -> {candidate} (L1 distance: {distance:.4f})")
```

Expected output:

```
Candidates for 'aple':
  -> apple (L1 distance: 1.4400)
  -> pineapple (L1 distance: 6.2160)
  -> peach (L1 distance: 6.6600)
Candidates for 'bannana':
  -> banana (L1 distance: 1.1349)
  -> orange (L1 distance: 9.1508)
  -> apple (L1 distance: 9.5592)
Candidates for 'orng':
  -> orange (L1 distance: 3.5833)
  -> banana (L1 distance: 8.4722)
  -> apple (L1 distance: 9.2500)
```

### How to save and load an index:
Building the FAISS index is the only slow part of the setup. To avoid waiting on every startup, you can build the index once, save it to disk, and load it instantly later. For exemple, the ~160k word dictionary used in the Birkbeck benchmark takes about 5 seconds to save and 0.75 seconds to load it back into memory.

```python
from vecfuzz import VecFuzz
vecfuzz = VecFuzz().build(["..."]).save("index.zip")

# later, or in another process
from vecfuzz import VecFuzz
vecfuzz = VecFuzz().load("index.zip")
```

## Contributing

Early-stage project. Issues and PRs are welcome, especially around closing the substitution-error gap.


## License

MIT. Use it, modify it, ship it. Keep the copyright notice.
