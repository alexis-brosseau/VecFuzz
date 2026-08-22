# VecFuzz: Vector-based Fuzzy Matching

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A **fast approximate string matching library** that turns words into compact vectors so you can find the closest match, even when the query is riddled with typos! It’s a fuzzy matching on steroids: sub‑millisecond lookup times and linear memory scaling.


## The trade-off

Fuzzy matching has a three-way tension between speed, memory, and accuracy. VecFuzz dominate on speed and memory, while remaining competitive on accuracy. What it does:

- **Memory stays flat** as typo-tolerance increases. SymSpell's index size grows combinatorially with max edit distance while VecFuzz grows linearly **O(N)**.
- **Query speed stays high** relative to brute-force comparison methods. RapidFuzz and raw Levenshtein score every candidate per query; VecFuzz searches an ANN index instead, ~100x+ faster in my tests.
- **Accuracy is strong**, including on real human misspellings (see the Birkbeck results below). Deletion-heavy errors are the hardest case, where the margin over SymSpell's higher-order configs is smallest.
- **Multi-threaded (via FAISS)** while SymSpell runs one query at a time. More cores widen VecFuzz's advantage, while SymSpell's lookup speed is mostly a function of single-core clock speed.

If your dictionary is large, your memory budget is tight, queries need to be fast, and you can tolerate a slower build, VecFuzz is a strong pick.


## How it works

Each word is converted into a fixed-length vector with four sub-components:

1. **Character frequency:** how often each letter appears.
2. **Preceding-position density:** for each character, the sum of normalized positions of all characters before it. This captures *how much of the word has already gone by*.
3. **Succeeding-position density:** for each character, the sum of normalized positions of all characters after it. This captures *how much of the word is still ahead*.
4. **Phase-encoded position:** a small sinusoidal (cos/sin) expansion of each character's position across a few frequency bands.
5. **Adjacency-hash:** a fixed-size hashed count of (character, next-character) bigrams. This captures local character order.

All sub-vectors are normalized by word length, so "apple" and "apples" land close together. The sub-vectors are concatenated and indexed with FAISS HNSW under Manhattan (L1) distance; a query is vectorized the same way and the nearest neighbours in the index become your top-k candidates.

**Known limitation:** none of these five components encode character *adjacency*, i.e. which specific character precedes or follows another. That's the main open gap for substitution-heavy workloads.

## Benchmark Highlights

### Synthetic edit-distance sweep

Dictionary of 100k words, compared against SymSpell at three delete-distance/prefix-length configs (d2/p7, d3/p9, d4/p12). Tested on a Xeon E5-2690 v4.

- **Recall@1 by error type and edit count**:
  - *Substitutions*: VecFuzz is close to parity with SymSpell d4/p12, leading at 1-2 edits, and trailing by only a few points at 3-5 edits.
  - *Transpositions*: VecFuzz starts near-perfect and stays well above all SymSpell configs at every edit count.
  - *Insertions*: SymSpell is slightly ahead but drop to 0 once edits exceed its configured max distance. VecFuzz degrades gracefully, still >50% recall at 9 insertion edits.
  - *Deletions*: Ahead of SymSpell at every edit count, but both degrade quickly past 2-3 edits.
![Accuracy by Error Type Chart](benchmark_outputs/accuracy_by_error_type_and_edits.png)

- **Lookup speed**:  SymSpell d2/p7 is fastest at small dictionaries, but VecFuzz overtakes it at larger dictionaries sizes when using more threads.
![Lookup Speed Chart](benchmark_outputs/lookup_speed_vs_vocab_size.png)

- **Memory**: VecFuzz grows roughly linearly and stays far below SymSpell d3/p9 and d4/p12 at every size tested. It is only slightly larger than SymSpell d2/p7 at 150k words.
![Memory Footprint Chart](benchmark_outputs/memory_footprint_vs_vocab_size.png)

- **Build time**: VecFuzz is slower to build than SymSpell d2/p7 and d3/p9 at every size tested, and is only slightly faster than SymSpell d4/p12.
![Build Time Chart](benchmark_outputs/build_time_vs_vocab_size.png)

### Real-world human errors (Birkbeck Spelling Error Corpus)

Dictionary ~160k words, non-synthetic human misspellings (includes phonetic errors, dysgraphia, multi-error handwriting slips). Tested on a Xeon E5-2690 v4.

| Method          | Recall@1     | Recall@5     | Recall@10     | Recall@25     | Recall@100     | Size         |
|-----------------|--------------|--------------|---------------|---------------|----------------|--------------|
| **VecFuzz**     | **36.32%**   | **54.94%**   | **61.37%**    | **68.82%**    | **77.91%**     | 221.11MB     |
| SymSpell d2/p7  | 34.05%       | 48.92%       | 51.94%        | 54.58%        | 57.70%         | **190.88MB** |
| RapidFuzz       | 32.64%       | 51.74%       | 58.54%        | 66.56%        | 76.67%         | N/A          |
| Levenshtein     | 28.10%       | 46.73%       | 54.20%        | 62.64%        | 72.35%         | N/A          |

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
  -> apple (L1 distance: 1.7406)
  -> pineapple (L1 distance: 6.6818)
  -> peach (L1 distance: 8.6489)
Candidates for 'bannana':
  -> banana (L1 distance: 1.2612)
  -> orange (L1 distance: 8.4829)
  -> apple (L1 distance: 9.8025)
Candidates for 'orng':
  -> orange (L1 distance: 4.8798)
  -> banana (L1 distance: 8.0867)
  -> apple (L1 distance: 10.7739)
```

To save and load an index:

```python
from vecfuzz import VecFuzz
vecfuzz = VecFuzz().build(["..."]).save("index.zip")

# later, or in another process
from vecfuzz import VecFuzz
vecfuzz = VecFuzz().load("index.zip")
```


## When to use this

- Large dictionaries where SymSpell-style precomputed edit indexes get too large to fit in memory.
- Interactive/high-QPS fuzzy search where brute-force comparison (RapidFuzz, raw Levenshtein) is too slow.


## When not to use this

- Small dictionaries where memory and speed aren't a constraint.
- Workloads with frequent dictionary updates.

## Contributing

Early-stage project. Issues and PRs are welcome, especially around closing the substitution-error gap.


## License

MIT. Use it, modify it, ship it. Keep the copyright notice.
