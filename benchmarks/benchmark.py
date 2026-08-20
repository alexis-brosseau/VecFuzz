from __future__ import annotations

from vecfuzz import VecFuzz, Vectorizer

import argparse
import signal
from pathlib import Path
import json
import random
from pathlib import Path
from time import perf_counter
from unicodedata import normalize
from spellchecker import SpellChecker
from symspellpy import SymSpell, Verbosity
import faiss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_EDIT_LEVELS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
TYPO_TYPES = ("substitution", "transposition", "insertion", "deletion")

VECFUZZ_INSTANCES = {
    "VecFuzz Both": VecFuzz(
        metric=faiss.METRIC_L1,
        vectorizers=[
            Vectorizer.forward_density,
            Vectorizer.backward_density
        ]
    ),
    "VecFuzz Density": VecFuzz(
        metric=faiss.METRIC_L1,
        vectorizers=[
            Vectorizer.density,
        ]
    ),
}

SYMSPELL_INSTANCES = {
    #"SymSpell d2/p7": SymSpell(max_dictionary_edit_distance=2, prefix_length=7),
    #"SymSpell d3/p9": SymSpell(max_dictionary_edit_distance=3, prefix_length=9),
    "SymSpell d4/p12": SymSpell(max_dictionary_edit_distance=4, prefix_length=12),
}

_stop_requested = False

def _plot_lines(ax, x_values, series, title, xlabel, ylabel) -> None:
    color_map = {
        "VecFuzz": "#1D4ED8",
        "SymSpell d2/p7": "#FCA5A5",
        "SymSpell d3/p9": "#EF4444",
        "SymSpell d4/p12": "#7F1D1D",
    }
    palette = [
        "#fb6640", 
        "#f8c421", 
        "#49cc5c", 
        "#2c7ce5", 
        "#b361ff", 
        "#f15de2"
        "#f82553", 
    ]
    for idx, (label, y_values) in enumerate(series):
        is_vecfuzz = label == "VecFuzz"
        ax.plot(
            x_values, y_values, marker="o",
            linewidth=1.5,
            markersize=4.5,
            label=label, color=color_map.get(label, palette[idx % len(palette)]),
            zorder=10 if is_vecfuzz else 2,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)


def plot(state: dict[str, object], k: int, output_dir: str = "benchmark_outputs") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    vec_acc = state["vecfuzz"]
    symspell_acc = state["symspell"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes_flat = list(axes.flat)

    for idx, error_type in enumerate(TYPO_TYPES):
        ax = axes_flat[idx]
        x_values = list(DEFAULT_EDIT_LEVELS)

        vec_series = []
        for label, acc in vec_acc.items():
            by_edit = acc["by_error_and_edits"][error_type]
            vec_series.append((
                label,
                [
                    (by_edit[str(e)][f"recall"] / by_edit[str(e)]["count"]) if by_edit[str(e)]["count"] else 0.0
                    for e in DEFAULT_EDIT_LEVELS
                ],
            ))

        symspell_series = []
        for label, acc in symspell_acc.items():
            by_edit = acc["by_error_and_edits"][error_type]
            symspell_series.append((
                label,
                [
                    (by_edit[str(e)][f"recall"] / by_edit[str(e)]["count"]) if by_edit[str(e)]["count"] else 0.0
                    for e in DEFAULT_EDIT_LEVELS
                ],
            ))

        _plot_lines(
            ax, x_values, vec_series + symspell_series,
            f"{error_type.capitalize()} errors", "Number of edits", f"Recall@{k} accuracy",
        )
        ax.set_ylim(0, 1)

    fig.suptitle(
        f"Recall@1 accuracy by error type and number of edits (Higher is better)",
        y=1.02, fontsize=14,
    )
    fig.tight_layout()
    path = out / "benchmark.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path

def load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
    
def save_json(path: Path, data: dict[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)  # atomic-ish swap, avoids corrupting the file if interrupted mid-write


def normalize_word(word: str) -> str:
    return normalize("NFD", word).lower().strip()

def load_vocabulary(min_length: int = 4, max_words: int = None, seed: int = 0) -> list[str]:
    freq_dict = SpellChecker().word_frequency.dictionary
    vocab = [normalize_word(w) for w in freq_dict.keys() if len(w) >= min_length]
    rng = random.Random(seed)
    rng.shuffle(vocab)
    if max_words is not None:
        vocab = vocab[:max_words]
    return vocab

def empty_accumulator() -> dict[str, object]:
    return {
        "overall": {"count": 0.0, "recall": 0.0},
        "by_error_type": {t: {"count": 0.0, "recall": 0.0} for t in TYPO_TYPES},
        "by_edits": {str(e): {"count": 0.0, "recall": 0.0} for e in DEFAULT_EDIT_LEVELS},
        "by_error_and_edits": {
            t: {str(e): {"count": 0.0, "recall": 0.0} for e in DEFAULT_EDIT_LEVELS} for t in TYPO_TYPES
        },
    }
    
def accumulate_accuracy(accumulator: dict[str, object], test_cases, predictions) -> None:
    """Adds new session's raw hit/count data into an existing accumulator, in place."""
    for case, candidates in zip(test_cases, predictions):
        target = str(case["target"])
        error_type = str(case["error_type"])
        edits = str(case["edits"])
        hit = 1 if target in candidates else 0

        accumulator["overall"]["count"] += 1
        accumulator["overall"]["recall"] += hit
        accumulator["by_error_type"][error_type]["count"] += 1
        accumulator["by_error_type"][error_type]["recall"] += hit
        accumulator["by_edits"][edits]["count"] += 1
        accumulator["by_edits"][edits]["recall"] += hit
        accumulator["by_error_and_edits"][error_type][edits]["count"] += 1
        accumulator["by_error_and_edits"][error_type][edits]["recall"] += hit

def apply_single_typo(word: str, typo_type: str, rng: random.Random) -> str:
    if not word:
        return word
    if typo_type == "substitution":
        if len(word) < 2:
            return word
        idx = rng.randrange(len(word))
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        choices = alphabet.replace(word[idx], "")
        return word[:idx] + rng.choice(choices or alphabet) + word[idx + 1:]
    if typo_type == "transposition":
        if len(word) < 2:
            return word
        idx = rng.randrange(len(word) - 1)
        chars = list(word)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)
    if typo_type == "deletion":
        if len(word) < 2:
            return word
        idx = rng.randrange(len(word))
        return word[:idx] + word[idx + 1:]
    if typo_type == "insertion":
        idx = rng.randrange(len(word) + 1)
        letter = rng.choice("abcdefghijklmnopqrstuvwxyz")
        return word[:idx] + letter + word[idx:]
    raise ValueError(f"Unknown typo type: {typo_type}")

def generate_error_cases(
    vocab: list[str], cases_per_combo: int, edit_levels: list[int], seed: int
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    cases: list[dict[str, object]] = []
    for typo_type in TYPO_TYPES:
        for edits in edit_levels:
            for _ in range(cases_per_combo):
                target = rng.choice(vocab)
                query = target
                for _ in range(edits):
                    query = apply_single_typo(query, typo_type, rng)
                cases.append({"query": query, "target": target, "error_type": typo_type, "edits": edits})
    rng.shuffle(cases)
    return cases

def build_vecfuzz(instance, vocab: list[str], num_threads: int | None = None):
    t0 = perf_counter()
    vf = instance.build(vocab)
    return vf, perf_counter() - t0

def lookup_vecfuzz(instance, queries, k: int = 1):
    results = instance.lookup(list(queries), k)
    return [[w for w, _d in c] for _q, c in results]

def build_symspell(instance, vocab: list[str], frequencies: dict[str, int]):
    t0 = perf_counter()
    for word in vocab:
        instance.create_dictionary_entry(word, max(1, frequencies.get(word, 1)))
    return instance, perf_counter() - t0

def lookup_symspell(instance, queries, k: int = 1):
    out = []
    for q in queries:
        sug = instance.lookup(q, Verbosity.CLOSEST)
        out.append([s.term for s in sug[:k]])
    return out

def _handle_sigint(signum, frame):
    global _stop_requested
    print("\n[ablation] Pause requested - finishing current session, then saving...", flush=True)
    _stop_requested = True

def run_benchmark(
    vocab_size: int,
    cases_per_edit_level: int,
    max_sessions: int = None,  # None = run until Ctrl+C
    seed: int = 0,
    k: int = 1,
    resume: bool = False,
    output_dir: str = "benchmark_outputs",
) -> dict[str, object]:
    signal.signal(signal.SIGINT, _handle_sigint)

    path = Path(f"{output_dir}/benchmark_state_{vocab_size}_k{k}.json")
    state = load_json(path) if resume else None

    vocab = load_vocabulary(max_words=vocab_size, seed=seed)
    frequencies = SpellChecker().word_frequency.dictionary
    subset = vocab[:vocab_size]

    if state is None:
        print(f"[benchmark] Starting fresh.", flush=True)
        state = {
            "vocab_size": vocab_size,
            "vocab_fingerprint": len(vocab),
            "seed_base": seed,
            "sessions_run": 0,
            "vecfuzz":  {label: empty_accumulator() for label in VECFUZZ_INSTANCES.keys()},
            "symspell": {label: empty_accumulator() for label in SYMSPELL_INSTANCES.keys()},
        }
    else:
        if state["vocab_size"] != vocab_size or state["vocab_fingerprint"] != len(subset):
            raise ValueError(
                "Resuming with a different vocab_size/vocab than the saved state. "
                "Accumulated accuracy would be measuring different things - "
                "use a fresh --state-path or match the original vocab_size."
            )
        print(f"[benchmark] Resuming from {path} - {state['sessions_run']} sessions so far.", flush=True)

    print("[benchmark] Building indexes...", flush=True)
    vecfuzz_instances = {label: build_vecfuzz(instance, subset)[0] for label, instance in VECFUZZ_INSTANCES.items()}
    symspell_instances = {label: build_symspell(instance, subset, frequencies)[0] for label, instance in SYMSPELL_INSTANCES.items()}
    
    print("[benchmark] Starting sessions...", flush=True)
    session = 0
    try:
        while max_sessions is None or session < max_sessions:
            if _stop_requested:
                break

            session_seed = state["seed_base"] + state["sessions_run"]
            per_combo = max(1, cases_per_edit_level)
            cases = generate_error_cases(subset, per_combo, DEFAULT_EDIT_LEVELS, session_seed)
            queries = [str(c["query"]) for c in cases]

            # VecFuzz Lookups
            for label, instance in vecfuzz_instances.items():
                vec_preds = lookup_vecfuzz(instance, queries, k)
                accumulate_accuracy(state["vecfuzz"][label], cases, vec_preds)

            # SymSpell Lookups
            for label, instance in symspell_instances.items():
                preds = lookup_symspell(instance, queries, k)
                accumulate_accuracy(state["symspell"][label], cases, preds)

            state["sessions_run"] += 1
            session += 1
            save_json(path, state)

            print(
                "\r"
                f"[benchmark] session {state['sessions_run']} done ",
                end="",
                flush=True,
            )
        else:
            path = plot(state, k, output_dir)
            print(f"[benchmark] Figure written to {path}")
    finally:
        pass

    print(f"[benchmark] Stopped after {state['sessions_run']} total sessions. State saved to {path}.")
    return state

def main() -> None:
    p = argparse.ArgumentParser(description="Resumable ablation benchmark for VecFuzz sub-vectors.")
    p.add_argument("--vocab-size", type=int, default=50_000)
    p.add_argument("--cases", type=int, default=1_000)
    p.add_argument("--max-sessions", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--max-words", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="Resume from existing state file if present.")
    p.add_argument("--plot", action="store_true", help="Just render the accuracy figure from existing state and exit.")
    p.add_argument("--output-dir", default="benchmark_outputs")
    args = p.parse_args()

    if args.plot:
        state = load_json(Path(f"{args.output_dir}/benchmark_state_{args.vocab_size}_k{args.k}.json"))
        if state is None:
            raise SystemExit(f"No state file at {args.output_dir} to plot.")
        path = plot(state, args.k, args.output_dir)
        print(f"[benchmark] Figure written to {path}")
        return

    run_benchmark(
        vocab_size=args.vocab_size,
        cases_per_edit_level=args.cases,
        max_sessions=args.max_sessions,
        seed=args.seed,
        k=args.k,
        resume=args.resume,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()