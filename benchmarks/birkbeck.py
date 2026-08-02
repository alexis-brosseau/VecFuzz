import os
from time import time
from unicodedata import normalize
from tqdm import tqdm
from spellchecker import SpellChecker
from symspellpy import SymSpell
from pympler import asizeof
from vecfuzz import VecFuzz
import numpy as np
import matplotlib.pyplot as plt
from time import time
from spellchecker import SpellChecker
from tqdm import tqdm
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein
from symspellpy import SymSpell, Verbosity
from unicodedata import normalize, combining, east_asian_width
from pympler import asizeof

# ---------------------------
# BASELINE METHODS
# ---------------------------

def candidates_levenshtein(query, vocab, k=100):
    results = process.extract(query, vocab, scorer=Levenshtein.distance, limit=k)
    return [match[0] for match in results]

def candidates_rapidfuzz(query, vocab, k=100):
    results = process.extract(query, vocab, scorer=fuzz.ratio, limit=k)
    return [match[0] for match in results]

def candidates_symspell(query, vocab, symspell_instance, k=100):
    """SymSpell: use its built-in lookup, return Recall@k candidates."""
    suggestions = symspell_instance.lookup(query, Verbosity.ALL, max_edit_distance=2)
    # suggestions are already sorted by (distance, frequency)
    return [s.term for s in suggestions[:k]]

def candidates_vecfuzz_batch(queries, vocab, vecfuzz_instance, k=100):
    """VecFuzz batched lookup. Returns list of candidate lists."""
    results = vecfuzz_instance.lookup(queries, k)
    return [[word for word, dist in res[1]] for res in results]


def load_birkbeck_dataset(filepath):
    test_cases = []
    targets = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) == 2:
                # birkbeck contains entries like "a_bit: abit", replacing underscores with spaces
                target = parts[0].strip().replace("_", " ").lower()
                targets.add(target)
                queries = parts[1].strip().split()
                for q in queries:
                    q = q.replace("_", " ").lower()
                    test_cases.append({"query": q, "target": target})
    return test_cases, targets

def evaluate_simple(method_func, name, test_cases, vocab, args=[], is_batched=False):
    recall1, recall5, recall10, recall25, recall100 = 0, 0, 0, 0, 0
    total = len(test_cases)
    
    t0 = time()
    
    if is_batched:
        print(f"Running {name} in batch mode...", end="\r", flush=True)
        queries = [tc["query"] for tc in test_cases]
        all_preds = method_func(queries, vocab, *args)
        
        for tc, preds in zip(test_cases, all_preds):
            target = tc["target"]
            if target in preds[:1]: recall1 += 1
            if target in preds[:5]: recall5 += 1
            if target in preds[:10]: recall10 += 1
            if target in preds[:25]: recall25 += 1
            if target in preds[:100]: recall100 += 1
    else:
        for tc in tqdm(test_cases, desc=name, leave=False):
            target = tc["target"]
            preds = method_func(tc["query"], vocab, *args)
            if target in preds[:1]: recall1 += 1
            if target in preds[:5]: recall5 += 1
            if target in preds[:10]: recall10 += 1
            if target in preds[:25]: recall25 += 1
            if target in preds[:100]: recall100 += 1
            
    t1 = time()
    duration = t1 - t0
    
    return {
        "recall1": recall1 / total,
        "recall5": recall5 / total,
        "recall10": recall10 / total,
        "recall25": recall25 / total,
        "recall100": recall100 / total,
        "time_sec": duration,
    }

def run_birkbeck_benchmark(save_to_file=False):
    data_path = os.path.join(os.path.dirname(__file__), "birkbeck.txt")
    print(f"Loading dataset from {data_path}...")
    test_cases, birkbeck_targets = load_birkbeck_dataset(data_path)
    print(f"Loaded {len(test_cases)} test cases from birkbeck dataset.")

    def format_table(headers, rows):
        def display_width(text):
            width = 0
            for char in text:
                if combining(char):
                    continue
                width += 2 if east_asian_width(char) in ("W", "F") else 1
            return width

        widths = [display_width(header) for header in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], display_width(cell))

        def render_row(row):
            padded_cells = []
            for idx, cell in enumerate(row):
                padded_cells.append(cell + (" " * (widths[idx] - display_width(cell))))
            return "| " + " | ".join(padded_cells) + " |"

        separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
        return render_row(headers), separator, [render_row(row) for row in rows]

    print("Filtering and normalizing vocabulary...")
    freq_dict = SpellChecker().word_frequency.dictionary
    
    filtered_dict = {}
    for w, freq in freq_dict.items():
        norm_w = normalize('NFD', w).lower()
        filtered_dict[norm_w] = freq
        
    # Ensure all birkbeck targets are securely in our vocabulary so it's a fair test for all algorithms
    for t in birkbeck_targets:
        norm_t = normalize('NFD', t).lower()
        if norm_t not in filtered_dict:
            filtered_dict[norm_t] = 100  # Give it an arbitrary frequency
            
    vocab = list(filtered_dict.keys())
    print(f"Using {len(vocab)} words for the benchmark vocabulary.")
    
    # Build SymSpell dictionary and measure build time
    print("Building SymSpell d2/p7 dictionary (preprocessing)...")
    t0_build = time()
    symspell_instance_d2 = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    for w in vocab:
        freq = filtered_dict.get(w, 1)
        symspell_instance_d2.create_dictionary_entry(w, max(1, freq))
    t1_build = time()
    symspell_build_time_d2 = t1_build - t0_build
    symspell_size_d2 = asizeof.asizeof(symspell_instance_d2) / (1024 * 1024)

    # Build VecFuzz index
    print("Building VecFuzz index (preprocessing)...")
    t0_vecfuzz = time()
    vecfuzz_instance = VecFuzz().build(vocab)
    t1_vecfuzz = time()
    vecfuzz_build_time = t1_vecfuzz - t0_vecfuzz
    vecfuzz_size = asizeof.asizeof(vecfuzz_instance) / (1024 * 1024)

    # Define methods to benchmark
    methods = [
        (candidates_symspell, "SymSpell d2/p7", [symspell_instance_d2], False),
        (candidates_vecfuzz_batch, "VecFuzz", [vecfuzz_instance], True),
        (candidates_rapidfuzz, "RapidFuzz", [], False),
        (candidates_levenshtein, "Levenshtein", [], False),
    ]

    print("\nStarting Benchmark on birkbeck dataset...")
    results = []
    for func, name, args, is_batched in methods:
        res = evaluate_simple(func, name, test_cases, vocab, args, is_batched)
        res["name"] = name
        
        if name == "SymSpell d2/p7":
            res["build_time"] = symspell_build_time_d2
            res["build_size"] = symspell_size_d2
        elif name == "VecFuzz":
            res["build_time"] = vecfuzz_build_time
            res["build_size"] = vecfuzz_size
        else:
            res["build_time"] = 0.0
            res["build_size"] = 0.0
            
        results.append(res)
        
    metrics_to_rank = [
        ('recall1', True), 
        ('recall5', True), 
        ('recall10', True), 
        ('recall25', True), 
        ('recall100', True), 
        ('time_sec', False)
    ]
    medals = {r['name']: {} for r in results}
    for key, higher_is_better in metrics_to_rank:
        sorted_res = sorted(results, key=lambda x: x[key], reverse=higher_is_better)
        for i, rank_medal in enumerate(['🥇', '🥈', '🥉']):
            if i < len(sorted_res):
                medals[sorted_res[i]['name']][key] = " " + rank_medal

    headers = ["Method", "Recall@1 (%)", "Recall@5 (%)", "Recall@10 (%)", "Recall@25 (%)", "Recall@100 (%)", "Duration (s)", "Build (s)", "Size (MB)"]
    rows = []
    for r in results:
        rows.append([
            r['name'],
            f"{r['recall1'] * 100:.2f}%{medals[r['name']].get('recall1', '')}",
            f"{r['recall5'] * 100:.2f}%{medals[r['name']].get('recall5', '')}",
            f"{r['recall10'] * 100:.2f}%{medals[r['name']].get('recall10', '')}",
            f"{r['recall25'] * 100:.2f}%{medals[r['name']].get('recall25', '')}",
            f"{r['recall100'] * 100:.2f}%{medals[r['name']].get('recall100', '')}",
            f"{r['time_sec']:.3f}s{medals[r['name']].get('time_sec', '')}",
            f"{r['build_time']:.3f}s" if r['build_time'] > 0 else "N/A",
            f"{r['build_size']:.2f}" if r['build_size'] > 0 else "N/A",
        ])

    header_row, separator_row, body_rows = format_table(headers, rows)

    print("\n" + header_row)
    print(separator_row)
    for row in body_rows:
        print(row)

    if save_to_file:
        with open("benchmark_outputs/birkbeck_results.md", "w", encoding="utf-8") as f:
            f.write("# Birkbeck Benchmark Results\n\n")
            f.write(header_row + "\n")
            f.write(separator_row + "\n")
            for row in body_rows:
                f.write(row + "\n")
            
        print("\nSaved benchmark data to benchmark_outputs/birkbeck_results.md")

if __name__ == "__main__":
    run_birkbeck_benchmark(save_to_file=True)
