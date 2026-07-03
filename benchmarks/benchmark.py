import random
import numpy as np
import matplotlib.pyplot as plt
from time import time
from spellchecker import SpellChecker
from tqdm import tqdm
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein, DamerauLevenshtein, JaroWinkler
from symspellpy import SymSpell, Verbosity
from vecfuzz import VecFuzz
from unicodedata import normalize, combining, east_asian_width
from pympler import asizeof

# ---------------------------
# TYPO GENERATORS
# ---------------------------

SUB_WEIGHT = 0.25
SWAP_WEIGHT = 0.25
DEL_WEIGHT = 0.25
INS_WEIGHT = 0.25

def get_pos_label(i, length):
    if length <= 3:
        return "middle"
    if i == 0 or i == 1:
        return "prefix"
    elif i >= length - 2:
        return "suffix"
    else:
        return "middle"

def typo_substitution(word):
    if len(word) < 2:
        return word, "middle"
    i = random.randint(0, len(word)-1)
    c = random.choice("abcdefghijklmnopqrstuvwxyz".replace(word[i], ""))
    return word[:i] + c + word[i+1:], get_pos_label(i, len(word))


def typo_swap(word):
    if len(word) < 2:
        return word, "middle"
    i = random.randint(0, len(word)-2)
    lst = list(word)
    lst[i], lst[i+1] = lst[i+1], lst[i]
    return "".join(lst), get_pos_label(i, len(word))


def typo_deletion(word):
    if len(word) < 2:
        return word, "middle"
    i = random.randint(0, len(word)-1)
    return word[:i] + word[i+1:], get_pos_label(i, len(word))


def typo_insertion(word):
    i = random.randint(0, len(word))
    c = random.choice("abcdefghijklmnopqrstuvwxyz")
    return word[:i] + c + word[i:], get_pos_label(i, len(word))


def generate_typos(vocab, n=5000, get_num_edits=lambda: 1):
    # realistic weights: Sub is most common, then Ins/Del, then Swap
    # You can adjust these weights
    types = [
        ("substitution", typo_substitution),
        ("swap", typo_swap),
        ("deletion", typo_deletion),
        ("insertion", typo_insertion)
    ]
    weights = [SUB_WEIGHT, SWAP_WEIGHT, DEL_WEIGHT, INS_WEIGHT]
    
    test_cases = []

    for _ in range(n):
        w = random.choice(vocab)
        t_name, t_func = random.choices(types, weights=weights, k=1)[0]
        
        num_edits = get_num_edits()
        
        current_w = w
        pos_label = "middle"
        for _ in range(num_edits):
            current_w, pos_label = t_func(current_w)

        test_cases.append({
            "query": current_w,
            "target": w,
            "error_type": t_name,
            "error_pos": pos_label,
            "edits": num_edits
        })

    return test_cases


# ---------------------------
# BASELINE METHODS
# ---------------------------

def candidates_levenshtein(query, vocab, k=5):
    results = process.extract(query, vocab, scorer=Levenshtein.distance, limit=k)
    return [match[0] for match in results]

def candidates_damerau_levenshtein(query, vocab, k=5):
    results = process.extract(query, vocab, scorer=DamerauLevenshtein.distance, limit=k)
    return [match[0] for match in results]

def candidates_jaro_winkler(query, vocab, k=5):
    results = process.extract(query, vocab, scorer=JaroWinkler.similarity, limit=k)
    return [match[0] for match in results]

def candidates_rapidfuzz(query, vocab, k=5):
    results = process.extract(query, vocab, scorer=fuzz.ratio, limit=k)
    return [match[0] for match in results]

def candidates_symspell(query, vocab, symspell_instance, k=5):
    """SymSpell: use its built-in lookup, return Recall@k candidates."""
    suggestions = symspell_instance.lookup(query, Verbosity.CLOSEST, max_edit_distance=2)
    # suggestions are already sorted by (distance, frequency)
    return [s.term for s in suggestions[:k]]

def candidates_bktree(query, vocab, bktree_instance, k=5):
    """BK-Tree: find words within edit distance 2, sort by distance."""
    results = bktree_instance.find(query, 2)
    # results is a list of (distance, word); sort by distance
    results.sort(key=lambda x: x[0])
    return [word for dist, word in results[:k]]

def candidates_vecfuzz_batch(queries, vocab, vecfuzz_instance, k=5):
    """VecFuzz batched lookup. Returns list of candidate lists."""
    results = vecfuzz_instance.lookup(queries, k)
    return [[word for word, dist in res[1]] for res in results]

def candidates_norvig(query, vocab, freq_dict, k=5):
    """Norvig's spelling corrector candidate generation."""
    def edits1(word):
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(word): 
        return (e2 for e1 in edits1(word) for e2 in edits1(e1))
        
    def known(words): 
        return set(w for w in words if w in freq_dict)
        
    cands = known([query]) or known(edits1(query)) or known(edits2(query)) or {query}
    return sorted(cands, key=lambda w: freq_dict.get(w, 0), reverse=True)[:k]


# ---------------------------
# EVALUATION
# ---------------------------

def evaluate_accuracy(method_func, name, test_cases, vocab, args=[], trial_info="", is_batched=False):
    top1 = 0
    top3 = 0
    top5 = 0
    total = len(test_cases)
    
    # Stratified metrics
    stats = {
        "error_type": {"substitution": {"count": 0, "top1": 0}, "swap": {"count": 0, "top1": 0}, "deletion": {"count": 0, "top1": 0}, "insertion": {"count": 0, "top1": 0}},
        "error_pos": {"prefix": {"count": 0, "top1": 0}, "middle": {"count": 0, "top1": 0}, "suffix": {"count": 0, "top1": 0}},
        "edits": {1: {"count": 0, "top1": 0}, 2: {"count": 0, "top1": 0}}
    }

    t0 = time()

    desc_str = f"{trial_info} - {name}" if trial_info else name

    if is_batched:
        print(f"Running {name} in batch mode...", end="\r", flush=True)
        queries = [tc["query"] for tc in test_cases]
        all_preds = method_func(queries, vocab, *args)

    for i, tc in enumerate(tqdm(test_cases, total=total, desc=desc_str, leave=False)):
        target = tc["target"]
        
        if is_batched:
            preds = all_preds[i]
        else:
            q = tc["query"]
            preds = method_func(q, vocab, *args) if args else method_func(q, vocab)

        is_top1 = False
        if target in preds[:1]:
            top1 += 1
            is_top1 = True
        if target in preds[:3]:
            top3 += 1
        if target in preds[:5]:
            top5 += 1
            
        # Update stratified stats
        e_type = tc["error_type"]
        e_pos = tc["error_pos"]
        e_edits = tc["edits"]
        
        if e_type in stats["error_type"]:
            stats["error_type"][e_type]["count"] += 1
            if is_top1: stats["error_type"][e_type]["top1"] += 1
            
        if e_pos in stats["error_pos"]:
            stats["error_pos"][e_pos]["count"] += 1
            if is_top1: stats["error_pos"][e_pos]["top1"] += 1
            
        if e_edits in stats["edits"]:
            stats["edits"][e_edits]["count"] += 1
            if is_top1: stats["edits"][e_edits]["top1"] += 1

    t1 = time()
    duration = t1 - t0

    return {
        "top1": top1 / total,
        "top3": top3 / total,
        "top5": top5 / total,
        "time_sec": duration,
        "iters_sec": total / duration if duration > 0 else 0.0,
        "stats": stats
    }


# ---------------------------
# RUN TEST
# ---------------------------

def run_benchmark(freq_dict, n_trials, n_per_trial, seed=0, save_to_file=False):
    vocab = list(freq_dict.keys())

    #Build SymSpell dictionary and measure build time (preprocessing)
    print("\nInitializing SymSpell dictionary (preprocessing)...")
    t0_build = time()
    
    symspell_instance = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    for w in vocab:
        freq = freq_dict.get(w, 1)
        symspell_instance.create_dictionary_entry(w, max(1, freq)) # Ensure at least a count of 1
    
    t1_build = time()
    symspell_build_time = t1_build - t0_build
    symspell_size = asizeof.asizeof(symspell_instance) / (1024 * 1024)

    # Build VecFuzz index
    print("\nBuilding VecFuzz index (preprocessing)...")
    t0_vecfuzz = time()
    vecfuzz_instance = VecFuzz().index(vocab)
    t1_vecfuzz = time()
    vecfuzz_build_time = t1_vecfuzz - t0_vecfuzz
    vecfuzz_size = asizeof.asizeof(vecfuzz_instance) / (1024 * 1024)

    # Define methods to benchmark
    methods = [
        (candidates_vecfuzz_batch, "VecFuzz", [vecfuzz_instance], True),
        (candidates_symspell, "SymSpell", [symspell_instance], False),
        (candidates_rapidfuzz, "RapidFuzz", [], False),
        (candidates_jaro_winkler, "Jaro-Winkler", [], False),
        (candidates_damerau_levenshtein, "Damerau-Levenshtein", [], False),
        (candidates_levenshtein, "Levenshtein", [], False),
        (candidates_norvig, "Norvig", [freq_dict], False),
    ]

    results = {name: {"top1": [], "top3": [], "top5": [], "time_sec": [], "iters_sec": [], "build_time": [], "build_size": [], "stats": []} for _, name, _, _ in methods}

    print("\nStarting Benchmark (", n_trials, "trials,", n_per_trial, "queries each)")

    for tr in range(n_trials):
        random.seed(seed + tr)
        np.random.seed(seed + tr)

        get_num_edits = lambda: random.choices([1, 2], weights=[0.8, 0.2], k=1)[0]
        test_cases = generate_typos(vocab, n_per_trial, get_num_edits)
        trial_str = f"Trial {tr + 1}/{n_trials}"

        for func, name, args, is_batched in methods:
            res = evaluate_accuracy(func, name, test_cases, vocab, args, trial_info=trial_str, is_batched=is_batched)
            for k in ("top1", "top3", "top5", "time_sec", "iters_sec"):
                results[name][k].append(res[k])
            results[name]["stats"].append(res["stats"])
            # record build time relevant to method
            if name.startswith("SymSpell"):
                results[name]["build_time"].append(symspell_build_time)
                results[name]["build_size"].append(symspell_size)
            elif name.startswith("VecFuzz"):
                results[name]["build_time"].append(vecfuzz_build_time)
                results[name]["build_size"].append(vecfuzz_size)
            else:
                results[name]["build_time"].append(0.0)
                results[name]["build_size"].append(0.0)

    # Print aggregated mean +/- std, include symspell build time separately
    def mean_std(arr):
        if not arr: return 0.0, 0.0
        a = np.asarray(arr)
        return a.mean(), a.std()

    # Calculate means for ranking
    aggregated_results = []
    for _, name, _, _ in methods:
        t1_mean, t1_std = mean_std(results[name]["top1"])
        t3_mean, t3_std = mean_std(results[name]["top3"])
        t5_mean, t5_std = mean_std(results[name]["top5"])
        time_mean, time_std = mean_std(results[name]["time_sec"])
        iters_mean, iters_std = mean_std(results[name]["iters_sec"])
        build_mean, _ = mean_std(results[name]["build_time"])
        size_mean, _ = mean_std(results[name]["build_size"])
        
        aggregated_results.append({
            "name": name,
            "top1": t1_mean, "top1_std": t1_std,
            "top3": t3_mean, "top3_std": t3_std,
            "top5": t5_mean, "top5_std": t5_std,
            "time_sec": time_mean, "time_sec_std": time_std,
            "iters_sec": iters_mean, "iters_sec_std": iters_std,
            "build_time": build_mean,
            "build_size": size_mean
        })

    metrics_to_rank = [
        ('top1', True), ('top3', True), ('top5', True),
        ('time_sec', False)
    ]
    medals = {r['name']: {} for r in aggregated_results}
    for key, higher_is_better in metrics_to_rank:
        sorted_res = sorted(aggregated_results, key=lambda x: x[key], reverse=higher_is_better)
        for i, rank_medal in enumerate(['🥇', '🥈', '🥉']):
            if i < len(sorted_res):
                medals[sorted_res[i]['name']][key] = " " + rank_medal

    # helper for aggregating stats across trials
    def agg_stats(name, category, key):
        t_counts = 0
        t_top1 = 0
        for trial_stat in results[name]["stats"]:
            t_counts += trial_stat[category][key]["count"]
            t_top1 += trial_stat[category][key]["top1"]
        return (t_top1 / t_counts * 100) if t_counts > 0 else 0.0

    method_names = [name for _, name, _, _ in methods]

    def rank_metric(metric_key, category, key, higher_is_better=True):
        ranked = sorted(method_names, key=lambda name: agg_stats(name, category, key), reverse=higher_is_better)
        for i, rank_medal in enumerate(['🥇', '🥈', '🥉']):
            if i < len(ranked):
                medals[ranked[i]][metric_key] = " " + rank_medal

    def format_table(headers, rows):
        def display_width(text):
            width = 0
            for char in text:
                if combining(char):
                    continue
                width += 2 if east_asian_width(char) in ("W", "F") else 1
            return width

        widths = [display_width(h) for h in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], display_width(cell))

        def render_row(row):
            rendered_cells = []
            for idx, cell in enumerate(row):
                pad = widths[idx] - display_width(cell)
                rendered_cells.append(cell + (" " * pad))
            return "| " + " | ".join(rendered_cells) + " |"

        separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
        return render_row(headers), separator, [render_row(row) for row in rows]

    # Rank every error-type and error-position column independently.
    rank_metric("error_type_substitution", "error_type", "substitution")
    rank_metric("error_type_insertion", "error_type", "insertion")
    rank_metric("error_type_deletion", "error_type", "deletion")
    rank_metric("error_type_swap", "error_type", "swap")
    rank_metric("edits_1", "edits", 1)
    rank_metric("edits_2", "edits", 2)
    rank_metric("error_pos_prefix", "error_pos", "prefix")
    rank_metric("error_pos_middle", "error_pos", "middle")
    rank_metric("error_pos_suffix", "error_pos", "suffix")

    error_type_headers = ["Method", "Substitution", "Insertion", "Deletion", "Transposition"]
    error_count_headers = ["Method", "1-Error", "2-Errors", "Prefix", "Middle", "Suffix"]

    error_type_rows = []
    error_count_rows = []
    for name in method_names:
        sub_acc = agg_stats(name, "error_type", "substitution")
        ins_acc = agg_stats(name, "error_type", "insertion")
        del_acc = agg_stats(name, "error_type", "deletion")
        swap_acc = agg_stats(name, "error_type", "swap")
        e1_acc = agg_stats(name, "edits", 1)
        e2_acc = agg_stats(name, "edits", 2)
        pref_acc = agg_stats(name, "error_pos", "prefix")
        mid_acc = agg_stats(name, "error_pos", "middle")
        suf_acc = agg_stats(name, "error_pos", "suffix")

        error_type_rows.append([
            name,
            f"{sub_acc:.1f}%{medals[name].get('error_type_substitution', '')}",
            f"{ins_acc:.1f}%{medals[name].get('error_type_insertion', '')}",
            f"{del_acc:.1f}%{medals[name].get('error_type_deletion', '')}",
            f"{swap_acc:.1f}%{medals[name].get('error_type_swap', '')}",
        ])

        error_count_rows.append([
            name,
            f"{e1_acc:.1f}%{medals[name].get('edits_1', '')}",
            f"{e2_acc:.1f}%{medals[name].get('edits_2', '')}",
            f"{pref_acc:.1f}%{medals[name].get('error_pos_prefix', '')}",
            f"{mid_acc:.1f}%{medals[name].get('error_pos_middle', '')}",
            f"{suf_acc:.1f}%{medals[name].get('error_pos_suffix', '')}",
        ])

    # Print in the requested three-table markdown format (no iters/sec column)
    print("\n#### Overall Accuracy and Speed\n")
    overall_headers = ["Method", "Recall@1 (%)", "Recall@3 (%)", "Recall@5 (%)", "Duration (s)", "Build (s)", "Size (MB)"]
    overall_rows = []
    for r in aggregated_results:
        overall_rows.append([
            r['name'],
            f"{r['top1'] * 100:.2f}%{medals[r['name']].get('top1', '')}",
            f"{r['top3'] * 100:.2f}%{medals[r['name']].get('top3', '')}",
            f"{r['top5'] * 100:.2f}%{medals[r['name']].get('top5', '')}",
            f"{r['time_sec']:.3f}s{medals[r['name']].get('time_sec', '')}",
            f"{r['build_time']:.3f}s" if r['build_time'] > 0 else "N/A",
            f"{r['build_size']:.2f}" if r['build_size'] > 0 else "N/A",
        ])

    overall_header, overall_separator, overall_body = format_table(overall_headers, overall_rows)
    print(overall_header)
    print(overall_separator)
    for row in overall_body:
        print(row)

    # Recall@1 by error type
    print("\n#### Top‑1 Accuracy by Error Type\n")
    type_header, type_separator, type_rows = format_table(error_type_headers, error_type_rows)
    print(type_header)
    print(type_separator)
    for row in type_rows:
        print(row)

    # Recall@1 by error count and position
    print("\n#### Top‑1 Accuracy by Error Count and Error Position\n")
    count_header, count_separator, count_rows = format_table(error_count_headers, error_count_rows)
    print(count_header)
    print(count_separator)
    for row in count_rows:
        print(row)

    if save_to_file:
        with open("benchmark_results.md", "w", encoding="utf-8") as f:
            f.write("#### Overall Accuracy and Speed\n\n")
            f.write(overall_header + "\n")
            f.write(overall_separator + "\n")
            for row in overall_body:
                f.write(row + "\n")

            f.write("\n#### Top‑1 Accuracy by Error Type\n\n")
            f.write(type_header + "\n")
            f.write(type_separator + "\n")
            for row in type_rows:
                f.write(row + "\n")

            f.write("\n#### Top‑1 Accuracy by Error Count and Error Position\n\n")
            f.write(count_header + "\n")
            f.write(count_separator + "\n")
            for row in count_rows:
                f.write(row + "\n")

        print("\nSaved benchmark data to benchmark_results.md")

# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    freq_dict = SpellChecker().word_frequency.dictionary
    
    print("Filtering and normalizing vocabulary...")
    
    # Normalize words to NFD form to ensure consistent character representation, and filter out very short words
    filtered_dict = {
        normalize('NFD', w): freq 
        for w, freq in freq_dict.items()
        if len(w) > 3
    }
    
    print(f"Using {len(filtered_dict)}/{len(freq_dict)} alphabetic words for the benchmark")
    
    run_benchmark(filtered_dict, n_trials=5, n_per_trial=5_000, seed=0, save_to_file=True)