from __future__ import annotations

import argparse
import signal
from pathlib import Path
from time import perf_counter
from typing import Dict
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from benchmark_common import (
    DEFAULT_EDIT_LEVELS, DEFAULT_SYMSPELL_CONFIGS, TYPO_TYPES, SymSpellConfig,
    accumulate_accuracy, build_symspell, build_vecfuzz, empty_accuracy_accumulator,
    generate_error_cases, load_json, load_vocabulary, lookup_symspell, lookup_vecfuzz, save_json,
)
from spellchecker import SpellChecker

_stop_requested = False

def _plot_lines(ax, x_values, series, title, xlabel, ylabel) -> None:
    color_map = {
        "VecFuzz": "#1D4ED8",
        "SymSpell d2/p7": "#FCA5A5",
        "SymSpell d3/p9": "#EF4444",
        "SymSpell d4/p12": "#7F1D1D",
    }
    palette = ["#0F766E", "#1D4ED8", "#B45309", "#7C3AED", "#DC2626"]
    for idx, (label, y_values) in enumerate(series):
        is_vecfuzz = label == "VecFuzz"
        ax.plot(
            x_values, y_values, marker="o",
            linewidth=2.5 if is_vecfuzz else 2,
            label=label, color=color_map.get(label, palette[idx % len(palette)]),
            zorder=10 if is_vecfuzz else 2,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)


def plot_accuracy(state: Dict[str, object], k: int, output_dir: str = "benchmark_outputs") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    vec_acc = state["vecfuzz"]
    symspell_acc = state["symspell"]  # dict keyed by label now, not a list — see note below

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes_flat = list(axes.flat)

    for idx, error_type in enumerate(TYPO_TYPES):
        ax = axes_flat[idx]
        x_values = list(DEFAULT_EDIT_LEVELS)

        vec_by_edit = vec_acc["by_error_and_edits"][error_type]
        vec_series = [(
            "VecFuzz",
            [
                (vec_by_edit[str(e)][f"recall@{k}"] / vec_by_edit[str(e)]["count"]) if vec_by_edit[str(e)]["count"] else 0.0
                for e in DEFAULT_EDIT_LEVELS
            ],
        )]

        symspell_series = []
        for label, acc in symspell_acc.items():
            by_edit = acc["by_error_and_edits"][error_type]
            symspell_series.append((
                label,
                [
                    (by_edit[str(e)][f"recall@{k}"] / by_edit[str(e)]["count"]) if by_edit[str(e)]["count"] else 0.0
                    for e in DEFAULT_EDIT_LEVELS
                ],
            ))

        _plot_lines(
            ax, x_values, vec_series + symspell_series,
            f"{error_type.capitalize()} errors", "Number of edits", f"Recall@{k} accuracy",
        )
        ax.set_ylim(0, 1)

    total_cases = int(vec_acc["overall"]["count"])
    fig.suptitle(
        f"Recall@1 accuracy by error type and number of edits (Higher is better) — n={total_cases} cases",
        y=1.02, fontsize=14,
    )
    fig.tight_layout()
    path = out / "accuracy_by_error_type_and_edits.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path

def _handle_sigint(signum, frame):
    global _stop_requested
    print("\n[accuracy] Pause requested — finishing current session, then saving...", flush=True)
    _stop_requested = True


def run_accuracy_benchmark(
    vocab_size: int = 150_000,
    cases_per_edit_level: int = 15_000,
    max_sessions: int = None,  # None = run until Ctrl+C
    seed: int = 0,
    k: int = 1,
    max_words: int = None,
    output_dir: str = "benchmark_outputs",
) -> Dict[str, object]:
    signal.signal(signal.SIGINT, _handle_sigint)

    path = Path(f"{output_dir}/accuracy_state_{vocab_size}_k{k}.json")
    state = load_json(path)

    vocab = load_vocabulary(max_words=max_words, seed=seed)
    frequencies = SpellChecker().word_frequency.dictionary
    configs = [SymSpellConfig(**c) for c in DEFAULT_SYMSPELL_CONFIGS]
    subset = vocab[:vocab_size]

    if state is None:
        print(f"[accuracy] No existing state at {path}, starting fresh.", flush=True)
        state = {
            "vocab_size": vocab_size,
            "vocab_fingerprint": len(subset),
            "seed_base": seed,
            "sessions_run": 0,
            "vecfuzz": empty_accuracy_accumulator(),
            "symspell": {c.label: empty_accuracy_accumulator() for c in configs},
        }
    else:
        if state["vocab_size"] != vocab_size or state["vocab_fingerprint"] != len(subset):
            raise ValueError(
                "Resuming with a different vocab_size/vocab than the saved state. "
                "Accumulated accuracy would be measuring different things - "
                "use a fresh --state-path or match the original vocab_size."
            )
        print(f"[accuracy] Resuming from {path} — {state['sessions_run']} sessions so far.", flush=True)

    print("[accuracy] Building indexes (once per run, not per session)...", flush=True)
    vecfuzz_index, _ = build_vecfuzz(subset)
    symspell_indexes = {c.label: build_symspell(subset, c, frequencies)[0] for c in configs}

    session = 0
    try:
        while max_sessions is None or session < max_sessions:
            if _stop_requested:
                break

            session_seed = state["seed_base"] + state["sessions_run"]
            per_combo = max(1, cases_per_edit_level)
            cases = generate_error_cases(subset, per_combo, DEFAULT_EDIT_LEVELS, session_seed)
            queries = [str(c["query"]) for c in cases]

            # VecFuzz: single batched call, no per-word loop -> no tqdm here.
            vec_preds = lookup_vecfuzz(vecfuzz_index, queries, k)
            accumulate_accuracy(state["vecfuzz"], cases, vec_preds)

            # SymSpell: lookup_symspell
            for cfg in configs:
                preds = lookup_symspell(symspell_indexes[cfg.label], queries, cfg.max_edit_distance, k)
                accumulate_accuracy(state["symspell"][cfg.label], cases, preds)

            state["sessions_run"] += 1
            session += 1
            save_json(path, state)

            total_cases = int(state["vecfuzz"]["overall"]["count"])
            recall = (
                state["vecfuzz"]["overall"]["recall1"] / total_cases if total_cases else 0.0
            )
            print(
                "\r"
                f"[accuracy] session {state['sessions_run']} done "
                f"| total cases: {total_cases} | vecfuzz recall@{k}: {recall:.4f}",
                end="",
                flush=True,
            )
    finally:
        pass

    print(f"[accuracy] Stopped after {state['sessions_run']} total sessions. State saved to {path}.")
    return state

def main() -> None:
    p = argparse.ArgumentParser(description="Resumable accuracy benchmark (VecFuzz vs SymSpell).")
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--cases", type=int, default=15_000)
    p.add_argument("--max-sessions", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--max-words", type=int, default=None)
    p.add_argument("--plot", action="store_true", help="Just render the accuracy figure from existing state and exit.")
    p.add_argument("--output-dir", default="benchmark_outputs")
    args = p.parse_args()

    if args.plot:
        state = load_json(Path(args.output_dir))
        if state is None:
            raise SystemExit(f"No state file at {args.output_dir} to plot.")
        path = plot_accuracy(state, args.k, args.output_dir)
        print(f"[accuracy] Figure written to {path}")
        return

    run_accuracy_benchmark(
        vocab_size=args.vocab_size,
        cases_per_edit_level=args.cases,
        max_sessions=args.max_sessions,
        seed=args.seed,
        k=args.k,
        max_words=args.max_words,
        output_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()