from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from time import perf_counter
from unicodedata import normalize

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spellchecker import SpellChecker

from vecfuzz import VecFuzz, Vectorizer


DEFAULT_MAX_THREADS = faiss.omp_get_max_threads() or 1

def _make_vecfuzz(vectorizers=Vectorizer.DEFAULT):
    return VecFuzz(vectorizers=vectorizers, metric=faiss.METRIC_L1)

VECFUZZ_INSTANCES = {
    "VecFuzz DEFAULT": _make_vecfuzz(vectorizers=Vectorizer.DEFAULT),
    "VecFuzz LITE": _make_vecfuzz(vectorizers=Vectorizer.LITE),
    
    # Ablation instances
    # "Frequency": _make_vecfuzz(vectorizers=[Vectorizer.frequency]),
    # "Density": _make_vecfuzz(vectorizers=[Vectorizer.density]),
    # "Postion Avg": _make_vecfuzz(vectorizers=[Vectorizer.position_avg]),
    # "Position Phase": _make_vecfuzz(vectorizers=[Vectorizer.position_phase]),
    # "Position RBF": _make_vecfuzz(vectorizers=[Vectorizer.position_rbf]),
    # "Bigram": _make_vecfuzz(vectorizers=[Vectorizer.bigram]),
    
    # Dual Ablation instances
    # "Freq + Density": _make_vecfuzz(vectorizers=[Vectorizer.frequency, Vectorizer.density]),
    # "Freq + Pos Avg": _make_vecfuzz(vectorizers=[Vectorizer.frequency, Vectorizer.position_avg]),
    # "Freq + Pos Phase": _make_vecfuzz(vectorizers=[Vectorizer.frequency, Vectorizer.position_phase]),
    # "Freq + Pos RBF": _make_vecfuzz(vectorizers=[Vectorizer.frequency, Vectorizer.position_rbf]),
    # "Freq + Bigram": _make_vecfuzz(vectorizers=[Vectorizer.frequency, Vectorizer.bigram]),
    # "Density + Pos Avg": _make_vecfuzz(vectorizers=[Vectorizer.density, Vectorizer.position_avg]),
    # "Density + Pos Phase": _make_vecfuzz(vectorizers=[Vectorizer.density, Vectorizer.position_phase]),
    # "Density + Pos RBF": _make_vecfuzz(vectorizers=[Vectorizer.density, Vectorizer.position_rbf]),
    # "Density + Bigram": _make_vecfuzz(vectorizers=[Vectorizer.density, Vectorizer.bigram]),
    # "Pos Avg + Pos Phase": _make_vecfuzz(vectorizers=[Vectorizer.position_avg, Vectorizer.position_phase]),
    # "Pos Avg + Pos RBF": _make_vecfuzz(vectorizers=[Vectorizer.position_avg, Vectorizer.position_rbf]),
    # "Pos Avg + Bigram": _make_vecfuzz(vectorizers=[Vectorizer.position_avg, Vectorizer.bigram]),
    # "Pos Phase + Pos RBF": _make_vecfuzz(vectorizers=[Vectorizer.position_phase, Vectorizer.position_rbf]),
    # "Pos Phase + Bigram": _make_vecfuzz(vectorizers=[Vectorizer.position_phase, Vectorizer.bigram]),
    # "Pos RBF + Bigram": _make_vecfuzz(vectorizers=[Vectorizer.position_rbf, Vectorizer.bigram]),
    
    # Triple Ablation instances
    # "Freq + Density + Pos Avg": _make_vecfuzz(vectorizers=[Vectorizer.frequency, Vectorizer.density, Vectorizer.position_avg]),
    # "Freq + Density + Pos Phase": _make_vecfuzz(vectorizers=[Vectorizer.frequency, Vectorizer.density, Vectorizer.position_phase]),
    # "Freq + Density + Pos RBF": _make_vecfuzz(vectorizers=[Vectorizer.frequency, Vectorizer.density, Vectorizer.position_rbf]),
    # "Freq + Density + Bigram": _make_vecfuzz(vectorizers=[Vectorizer.frequency, Vectorizer.density, Vectorizer.bigram]),
    # "Freq + Pos Avg + Pos Phase": _make_vecfuzz(vectorizers=[Vectorizer.frequency, Vectorizer.position_avg, Vectorizer.position_phase]),
    # "Freq + Pos Avg + Pos RBF": _make_vecfuzz(vectorizers=[Vectorizer.frequency, Vectorizer.position_avg, Vectorizer.position_rbf]),
    # "Freq + Pos Avg + Bigram": _make_vecfuzz(vectorizers=[Vectorizer.frequency, Vectorizer.position_avg, Vectorizer.bigram]),
    # "Freq + Pos Phase + Pos RBF": _make_vecfuzz(vectorizers=[Vectorizer.frequency, Vectorizer.position_phase, Vectorizer.position_rbf]),
    # "Freq + Pos Phase + Bigram": _make_vecfuzz(vectorizers=[Vectorizer.frequency, Vectorizer.position_phase, Vectorizer.bigram]),
    # "Freq + Pos RBF + Bigram": _make_vecfuzz(vectorizers=[Vectorizer.frequency, Vectorizer.position_rbf, Vectorizer.bigram]),
    # "Density + Pos Avg + Pos Phase": _make_vecfuzz(vectorizers=[Vectorizer.density, Vectorizer.position_avg, Vectorizer.position_phase]),
    # "Density + Pos Avg + Pos RBF": _make_vecfuzz(vectorizers=[Vectorizer.density, Vectorizer.position_avg, Vectorizer.position_rbf]),
    # "Density + Pos Avg + Bigram": _make_vecfuzz(vectorizers=[Vectorizer.density, Vectorizer.position_avg, Vectorizer.bigram]),
    # "Density + Pos Phase + Pos RBF": _make_vecfuzz(vectorizers=[Vectorizer.density, Vectorizer.position_phase, Vectorizer.position_rbf]),
    # "Density + Pos Phase + Bigram": _make_vecfuzz(vectorizers=[Vectorizer.density, Vectorizer.position_phase, Vectorizer.bigram]),
    # "Density + Pos RBF + Bigram": _make_vecfuzz(vectorizers=[Vectorizer.density, Vectorizer.position_rbf, Vectorizer.bigram]),
    # "Pos Avg + Pos Phase + Pos RBF": _make_vecfuzz(vectorizers=[Vectorizer.position_avg, Vectorizer.position_phase, Vectorizer.position_rbf]),
    # "Pos Avg + Pos Phase + Bigram": _make_vecfuzz(vectorizers=[Vectorizer.position_avg, Vectorizer.position_phase, Vectorizer.bigram]),
    # "Pos Avg + Pos RBF + Bigram": _make_vecfuzz(vectorizers=[Vectorizer.position_avg, Vectorizer.position_rbf, Vectorizer.bigram]),
    # "Pos Phase + Pos RBF + Bigram": _make_vecfuzz(vectorizers=[Vectorizer.position_phase, Vectorizer.position_rbf, Vectorizer.bigram]),
}


def clone_vecfuzz(instance: VecFuzz, threads: int) -> VecFuzz:
    benchmark_instance = copy.copy(instance)
    benchmark_instance.threads = threads
    benchmark_instance.entries = None
    benchmark_instance.vectors = None
    benchmark_instance.index = None
    return benchmark_instance


def load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def normalize_word(word: str) -> str:
    return normalize("NFD", word).lower().strip()


def load_vocabulary(min_length: int = 4, max_words: int | None = None, seed: int = 0) -> list[str]:
    frequency_dictionary = SpellChecker().word_frequency.dictionary
    vocabulary = [
        normalize_word(word)
        for word in frequency_dictionary.keys()
        if len(word) >= min_length
    ]
    random.Random(seed).shuffle(vocabulary)
    if max_words is not None:
        vocabulary = vocabulary[:max_words]
    return vocabulary


def make_queries(vocabulary: list[str], count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    queries = []
    for _ in range(max(1, count)):
        word = rng.choice(vocabulary)
        if len(word) > 1:
            index = rng.randrange(len(word))
            replacement = rng.choice("abcdefghijklmnopqrstuvwxyz".replace(word[index], "") or "abcdefghijklmnopqrstuvwxyz")
            word = word[:index] + replacement + word[index + 1:]
        queries.append(word)
    return queries


def empty_measurement() -> dict[str, float]:
    return {"count": 0.0, "total_seconds": 0.0}


def add_measurement(measurement: dict[str, float], elapsed: float) -> None:
    measurement["count"] += 1
    measurement["total_seconds"] += elapsed


def average_seconds(measurement: dict[str, float]) -> float:
    count = measurement["count"]
    return measurement["total_seconds"] / count if count else 0.0


def plot(state: dict[str, object], k: int, output_dir: str = "benchmark_outputs") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    thread_counts = [int(value) for value in state["thread_counts"]]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"VecFuzz DEFAULT": "#1D4ED8", "VecFuzz LITE": "#B91C1C"}
    for label in VECFUZZ_INSTANCES:
        build_seconds = [
            average_seconds(state["measurements"][label][str(threads)]["build"])
            for threads in thread_counts
        ]
        lookup_seconds = [
            average_seconds(state["measurements"][label][str(threads)]["lookup"])
            for threads in thread_counts
        ]
        axes[0].plot(
            thread_counts, build_seconds, marker="o", label=label, color=colors[label]
        )
        axes[1].plot(
            thread_counts, lookup_seconds, marker="o", label=label, color=colors[label]
        )
    axes[0].set_title("Build time")
    axes[0].set_ylabel("Average time (seconds)")
    axes[1].set_title("Lookup time")
    axes[1].set_ylabel("Average time (seconds)")
    for axis in axes:
        axis.set_xlabel("FAISS threads")
        axis.set_xticks(thread_counts)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("VecFuzz compute time by number of threads (Lower is better)")
    figure.tight_layout()

    path = output_path / f"benchmark_speed_k{k}.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def run_benchmark(
    vocab_size: int,
    cases_per_edit_level: int,
    max_sessions: int | None = 10,
    seed: int = 0,
    k: int = 1,
    resume: bool = False,
    output_dir: str = "benchmark_outputs",
    max_threads: int = DEFAULT_MAX_THREADS,
) -> dict[str, object]:
    if max_threads < 1:
        raise ValueError("max_threads must be at least 1")

    output_path = Path(output_dir)
    state_path = output_path / f"benchmark_speed_{vocab_size}_k{k}.json"
    state = load_json(state_path) if resume else None
    vocabulary = load_vocabulary(max_words=vocab_size, seed=seed)
    subset = vocabulary[:vocab_size]
    thread_counts = list(range(1, max_threads + 1))

    if state is None:
        state = {
            "vocab_size": vocab_size,
            "vocab_fingerprint": len(subset),
            "seed_base": seed,
            "max_threads": max_threads,
            "thread_counts": thread_counts,
            "sessions_run": 0,
            "measurements": {
                label: {
                    str(threads): {
                        "build": empty_measurement(),
                        "lookup": empty_measurement(),
                    }
                    for threads in thread_counts
                }
                for label in VECFUZZ_INSTANCES
            },
        }
    else:
        if (
            state["vocab_size"] != vocab_size
            or state["vocab_fingerprint"] != len(subset)
            or state["thread_counts"] != thread_counts
            or set(state["measurements"]) != set(VECFUZZ_INSTANCES)
        ):
            raise ValueError(
                "Cannot resume with a different vocabulary or max_threads. "
                "Use a fresh output directory or match the original arguments."
            )
        print(
            f"[benchmark] Resuming from {state_path} - "
            f"{state['sessions_run']} sessions so far.",
            flush=True,
        )

    print("[benchmark] Starting VecFuzz speed sessions...", flush=True)
    session = 0
    while (max_sessions is None or session < max_sessions):
        session_seed = state["seed_base"] + state["sessions_run"]
        queries = make_queries(subset, cases_per_edit_level, session_seed)

        for label, instance in VECFUZZ_INSTANCES.items():
            for threads in thread_counts:
                vecfuzz = clone_vecfuzz(instance, threads)
                start = perf_counter()
                vecfuzz.build(subset)
                add_measurement(
                    state["measurements"][label][str(threads)]["build"],
                    perf_counter() - start,
                )

                start = perf_counter()
                vecfuzz.lookup(queries, k)
                add_measurement(
                    state["measurements"][label][str(threads)]["lookup"],
                    perf_counter() - start,
                )

        state["sessions_run"] += 1
        session += 1
        save_json(state_path, state)
        print(f"\r[benchmark] session {state['sessions_run']} done", end="", flush=True)

    print()
    plot_path = plot(state, k, output_dir)
    print(f"[benchmark] Results saved to {state_path}")
    print(f"[benchmark] Figure written to {plot_path}")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable VecFuzz multi-thread speed benchmark.")
    parser.add_argument("--vocab-size", type=int, default=150_000)
    parser.add_argument("--cases", type=int, default=15_000, help="Number of batched lookup queries per session.")
    parser.add_argument("--max-sessions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--max-threads", type=int, default=20)
    parser.add_argument("--resume", action="store_true", help="Resume from existing state file if present.")
    parser.add_argument("--plot", action="store_false", help="Render the existing state file and exit.")
    parser.add_argument("--output-dir", default="benchmark_outputs")
    args = parser.parse_args()

    state_path = Path(args.output_dir) / f"benchmark_speed_{args.vocab_size}_k{args.k}.json"
    if args.plot:
        state = load_json(state_path)
        if state is None:
            raise SystemExit(f"No state file at {state_path} to plot.")
        path = plot(state, args.output_dir)
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
        max_threads=args.max_threads,
    )


if __name__ == "__main__":
    main()
