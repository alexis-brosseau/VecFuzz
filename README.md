# VecFuzz: Vector-based Fuzzy Search

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A **fast fuzzy search library** that turns words into compact vectors so you can find the closest match, even when the query is riddled with typos! It delivers high recall on human misspellings while keeping memory usage flat and having sub-millisecond lookup speeds.


## The trade-off

Fuzzy matching has a three-way tension between speed, memory, and accuracy. VecFuzz is designed to be competitive on all three:

*   **Memory**: SymSpell’s index size explodes combinatorially $O(N \cdot L^d)$ based on the max word length ($L$) and max edit distance ($d$). VecFuzz scales strictly linearly $O(N)$ with your dictionary size, completely independent of typo tolerance. Memory stays flat no matter how fuzzy you need to get.
*   **Build Speed**: SymSpell has to generate and hash every possible delete variant, leading to massive build times $O(N \cdot L^d)$ at higher edit distances ($d$). VecFuzz vectorizes the corpus and builds a FAISS HNSW graph, operating in $O(N \log N)$ time.
*   **Lookup Speed**: SymSpell achieves $O(1)$ hash lookups while VecFuzz traverses an HNSW graph in $O(\log N)$ time, but because it uses FAISS, it is **highly parallelizable** and scales down with additional threads.

VecFuzz is worth considering when if your dictionary is large, your memory budget is tight, queries need to be fast, and you can tolerate a slower build.


## How it works

Instead of a rigid, hardcoded pipeline, VecFuzz is built on a **composable vectorizer architecture**. 

When you initialize VecFuzz, you can pass a custom list of mathematical feature extractors via the `vectorizers` parameter. By default, it uses a highly optimized combination, but you can mix and match them to tune the model for your specific error types.

You can make your own vectorizer using the `@vectorizer` decorator or use some of the premade ones :
1.  **Length**: The length of the string as a single scalar feature.
2.  **N-grams**: A fixed-size hashed count of contiguous character n-grams to capture local ordering. The default configuration uses a bigram-style window size, and the hashed dimensionality defaults to 192, this was chosen by observing that recall gains diminish quickly past this point.
3.  **Bernstein**: Represents how each character is distributed across a word using Bernstein basis polynomials. The degree controls the resolution of the representation, and specific basis polynomials can optionally be selected with `indices`.
4.  **Phase-encoded Position**: A sinusoidal (cos/sin) expansion across frequency bands to capture global positional awareness. The default frequency bands were chosen by sweeping candidate values and keeping the set that maximized average recall across all four typo types.
5.  **Radial Basis Function (RBF)**: A smooth, local Gaussian positional encoding. As with the phase encoding, the number and spacing of Gaussian centers were tuned by sweeping values and selecting the configuration with the best average recall across the four typo types.
6.  **Frequency**: A log-frequency feature for dictionary entries. Omitted frequencies default to uniform values.

The sub-vectors are concatenated and indexed with FAISS HNSW under Manhattan (L1) distance; a query is vectorized the same way and the nearest neighbours in the index become the top-k candidates.

### The DEFAULT configuration

The DEFAULT configuration uses **hashed 2-grams + L1-normalized degree-0 Bernstein + the first and last degree-2 Bernstein polynomials + frequency weighted by 0.25**. The normalized degree-0 component is equivalent to character frequency (Bag-of-Characters), while the two degree-2 components capture character density at the beginning and end of the word. This configuration was chosen  because it gives recall close to the best of any tested combination while keeping the vector short, and therefore the index fast to build and query.

### The LITE configuration

The LITE configuration only uses **all degree-2 Bernstein basis polynomials + frequency weighted by 0.175**. It achieves recall close to DEFAULT while being roughly **2x smaller and faster**.


## Benchmark Highlights

### Real-world human errors (Birkbeck Spelling Error Corpus)
Tested on a ~160k word dictionary using 36k non-synthetic human misspellings (phonetic errors, dysgraphia, multi-error handwriting slips) on an Intel Xeon E5-2699 v4.
This error corpus was tested using word frequencies from the dictionary (pyspellchecker). The RapidFuzz and Levenshtein baselines received a frequency-sorted vocabulary, but their scoring functions remain unchanged.

#### 1. Recall Accuracy

| Method | Recall@1 | Recall@5 | Recall@10 | Recall@25 | Recall@100 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| VecFuzz DEFAULT | **42.69%**| **62.59%**| **68.41%**| **75.01%**| **82.82%** |
| VecFuzz LITE    | 39.58%    | 58.95%    | 64.78%    | 71.23%    | 79.04% |
| SymSpell d4/p12 | 38.50%    | 58.60%    | 64.36%    | 70.23%    | 77.65% |
| SymSpell d3/p9  | 37.62%    | 56.38%    | 61.29%    | 65.95%    | 71.34% |
| Levenshtein     | 37.31%    | 57.74%    | 63.88%    | 70.40%    | 79.13% |
| RapidFuzz       | 35.77%    | 54.89%    | 61.45%    | 69.50%    | 79.09% |
| SymSpell d2/p7  | 34.02%    | 48.88%    | 51.90%    | 54.57%    | 57.69% |

#### 2. Performance & Footprint

**For fairness, VecFuzz is run on a single thread.** In practice VecFuzz's build and lookup are both highly parallelizable and scale down with additional cores. See [Speed & Threading](#speed--threading).

| Method | Duration (s) | Build (s) | Size (MB) |
| :--- | :--- | :--- | :--- |
| VecFuzz LITE    | **7.552s** | 68.578s    | **145.06** |
| VecFuzz DEFAULT | 10.576s    | 141.115s   | 262.86     |
| SymSpell d2/p7  | 17.421s    | **2.955s** | 190.88     |
| SymSpell d3/p9  | 142.215s   | 12.457s    | 842.84     |
| RapidFuzz       | 554.347s   | N/A        | N/A        |
| Levenshtein     | 685.318s   | N/A        | N/A        |
| SymSpell d4/p12 | 765.296s   | 48.932s    | 3568.23    |

### Speed & Threading

Both build and lookup use FAISS's **multi-threading**, with compute time following an approximate inverse curve of the form $f(x) = 1/x + C$ after scaling, where $x$ is the number of threads and $C$ is the fixed overhead. Additional threads reduce runtime, with diminishing returns as the curve approaches $C$. The benchmarks below measure build time on a **150k-word dictionary** and lookup time for **15k queries** against that index. Tested on an Intel Xeon E5-2699 v4.

![VecFuzz compute time by number of threads](benchmark_outputs/benchmark_speed_k1.png)

### Synthetic edit-distance sweep

Dictionary of 150k words, compared against SymSpell at three delete-distance/prefix-length configs (d2/p7, d3/p9, d4/p12).

![Accuracy by Error Type Chart](benchmark_outputs/benchmark_accuracy_k1.png)

## When to use this
*   **Large dictionaries** where SymSpell-style precomputed edit indexes get too large to fit in memory.
*   **Interactive/high-QPS fuzzy search** where brute-force comparison (RapidFuzz, raw Levenshtein) is too slow.
*   **Multi-core environments** where you can leverage FAISS's multi-threading.

## When not to use this
*   **Small dictionaries** where memory and speed aren't a constraint.
*   **Workloads with frequent dictionary updates** (updating the HNSW graph takes longer than updating a SymSpell hash map).

## Installation

No pip package yet, the project is under active development. Import it directly as `from vecfuzz import VecFuzz` after placing `vecfuzz.py` in your working directory.
Clone or download this repository, make sure you have Python 3.10 or newer, and install the required dependencies.

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

## Contributing

Early-stage project. Issues and PRs are welcome, especially around closing the substitution-error gap.


## License

MIT. Use it, modify it, ship it. Keep the copyright notice.
