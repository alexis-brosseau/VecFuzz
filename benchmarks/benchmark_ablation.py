from __future__ import annotations

import argparse
import signal
from pathlib import Path
from typing import Dict, Sequence
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vecfuzz import VecFuzz

from benchmark_common import (
    DEFAULT_EDIT_LEVELS, TYPO_TYPES,
    accumulate_accuracy, empty_accuracy_accumulator,
    generate_error_cases, load_json, load_vocabulary, lookup_vecfuzz, save_json,
)

_stop_requested = False

DEFAULT_ABLATION_CONFIGS: dict[str, list[str]] = {
    "full":     ["frq", "pre", "suc", "phase", "adj"],
    
    "frq":          ["frq"],
    "pre-suc":      ["pre", "suc"],
    "phase":        ["phase"],
    "adj":          ["adj"],
    
    "frq pre-suc":      ["frq", "pre", "suc"],
    "frq phase":        ["frq", "phase"],
    "frq adj":          ["frq", "adj"],
    "pre-suc phase":    ["pre", "suc", "phase"],
    "pre-suc adj":      ["pre", "suc", "adj"],
    "phase adj":        ["phase", "adj"],
    
    "frq pre-suc adj":      ["frq", "pre", "suc", "adj"],
    "frq pre-suc phase":    ["frq", "pre", "suc", "phase"],
    "frq phase adj":        ["frq", "phase", "adj"],
    "pre-suc phase adj":    ["pre", "suc", "phase", "adj"],
}

PLOT_GROUPS: list[list[str]] = [
    ["frq", "pre-suc", "phase", "adj"],
    ["frq pre-suc", "frq phase", "frq adj", "pre-suc phase", "pre-suc adj", "phase adj"],
    ["frq pre-suc adj", "frq pre-suc phase", "frq phase adj", "pre-suc phase adj"],
]

def get_block_slices(vf: VecFuzz) -> dict[str, tuple[int, int]]:
    """
    Returns the (start, end) column range of each sub-vector block within the
    full vectorize_batch() output, in the same order they're concatenated:
    [frq, pre, suc, phase, adj].
    """
    c = vf._chars_len
    adj_dim = 64
    num_bands = 2
    offsets = {}
    start = 0
    for name, size in [
        ("frq", c),
        ("pre", c),
        ("suc", c),
        ("phase", 2 * num_bands * c),
        ("adj", adj_dim),
    ]:
        offsets[name] = (start, start + size)
        start += size
    return offsets


def make_ablation_vectorize_batch(vf: VecFuzz, keep: Sequence[str]):
    """
    Builds a drop-in replacement for vf.vectorize_batch that computes the full
    5-sub-vector representation once, then keeps only the blocks named in
    `keep` (any subset of: 'frq', 'pre', 'suc', 'phase', 'adj').

    Dimensionality shrinks to match the kept blocks, so build time, query
    time, and index size for each ablation variant reflect that variant's
    real cost -- not just its accuracy.

    Usage:
        vf = VecFuzz()
        vf.vectorize_batch = make_ablation_vectorize_batch(vf, ["pre", "suc"])
        vf.build(entries)  # now only pre+suc dims are computed and indexed
    """
    original = VecFuzz.vectorize_batch  # unbound -- captured once, safe to reuse
    offsets = get_block_slices(vf)
    keep_ranges = [offsets[name] for name in keep]

    def _ablated(words: list[str]) -> np.ndarray:
        full = original(vf, words)
        return np.concatenate([full[:, a:b] for a, b in keep_ranges], axis=1)

    return _ablated

def plot_all_groups(
    state: Dict[str, object],
    k: int,
    output_dir: str = "benchmark_outputs",
    groups: list[list[str]] = PLOT_GROUPS,
) -> list[Path]:
    """
    Renders one figure per tier in `groups`, each showing 'full' plus that
    tier's configs. Filenames are ablation_tier{N}.png in tier order.
    """
    paths = []
    for group_idx, group_labels in enumerate(groups, start=1):
        labels = ["full"] + list(group_labels)
        path = plot_accuracy(
            state, k, output_dir,
            subset=labels,
            filename=f"ablation_g{group_idx}.png",
        )
        paths.append(path)
        print(f"[ablation] Group {group_idx} ({len(group_labels)} configs + full) written to {path}")
    return paths


def _plot_lines(ax, x_values, series, title, xlabel, ylabel) -> None:
    palette = ["#1D4ED8", "#0F766E", "#B45309", "#7C3AED", "#DC2626", "#DB2777", "#65A30D"]
    for idx, (label, y_values) in enumerate(series):
        is_full = label.startswith("Full")
        ax.plot(
            x_values, y_values, marker="o",
            linewidth=2.0 if is_full else 1.3,
            markersize=4.5,
            label=label, color=palette[idx % len(palette)],
            zorder=10 if is_full else 2,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def plot_accuracy(
    state: Dict[str, object],
    k: int,
    output_dir: str = "benchmark_outputs",
    subset: Sequence[str] | None = None,
    filename: str | None = None,
) -> Path:
    """
    Renders the recall@k-by-error-type-and-edits figure.

    Args:
        subset: Optional list of ablation labels to include in the figure. If None, all configs are plotted.
        filename: output filename; defaults based on whether a subset was used.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    available = state["ablations"]
    labels = list(subset) if subset is not None else list(available.keys())
    missing = [l for l in labels if l not in available]
    if missing:
        raise ValueError(f"Requested labels not present in state: {missing}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes_flat = list(axes.flat)

    for idx, error_type in enumerate(TYPO_TYPES):
        ax = axes_flat[idx]
        x_values = list(DEFAULT_EDIT_LEVELS)

        series = []
        for label in labels:
            by_edit = available[label]["by_error_and_edits"][error_type]
            series.append((
                label,
                [
                    (by_edit[str(e)][f"recall{k}"] / by_edit[str(e)]["count"]) if by_edit[str(e)]["count"] else 0.0
                    for e in DEFAULT_EDIT_LEVELS
                ],
            ))

        _plot_lines(
            ax, x_values, series,
            f"{error_type.capitalize()} errors", "Number of edits", f"Recall@{k} accuracy",
        )
        ax.set_ylim(0, 1)

    fig.suptitle(f"VecFuzz ablation - recall@{k} by error type and edits", y=1.02, fontsize=14)
    fig.tight_layout()

    default_name = "ablation_singles.png" if subset is not None else "ablation_all.png"
    path = out / (filename or default_name)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _handle_sigint(signum, frame):
    global _stop_requested
    print("\n[ablation] Pause requested - finishing current session, then saving...", flush=True)
    _stop_requested = True


def _build_ablation_index(subset: list[str], keep: Sequence[str], num_threads: int) -> VecFuzz:
    vf = VecFuzz(num_threads=num_threads)
    vf.vectorize_batch = make_ablation_vectorize_batch(vf, keep)
    vf.build(subset)
    return vf


def run_ablation_benchmark(
    vocab_size: int = 150_000,
    cases_per_edit_level: int = 15_000,
    max_sessions: int | None = None,
    seed: int = 0,
    k: int = 1,
    max_words: int | None = None,
    output_dir: str = "benchmark_outputs",
    ablation_configs: dict[str, list[str]] | None = None,
) -> Dict[str, object]:
    signal.signal(signal.SIGINT, _handle_sigint)
    configs = ablation_configs or DEFAULT_ABLATION_CONFIGS

    path = Path(f"{output_dir}/ablation_state_{vocab_size}_k{k}.json")
    state = load_json(path)

    vocab = load_vocabulary(max_words=max_words, seed=seed)
    subset = vocab[:vocab_size]

    if state is None:
        print(f"[ablation] No existing state at {path}, starting fresh.", flush=True)
        state = {
            "vocab_size": vocab_size,
            "vocab_fingerprint": len(subset),
            "seed_base": seed,
            "sessions_run": 0,
            "ablations": {label: empty_accuracy_accumulator() for label in configs},
        }
    else:
        if state["vocab_size"] != vocab_size or state["vocab_fingerprint"] != len(subset):
            raise ValueError(
                "Resuming with a different vocab_size/vocab than the saved state. "
                "Accumulated accuracy would be measuring different things - "
                "use a fresh --state-path or match the original vocab_size."
            )
        print(f"[ablation] Resuming from {path} - {state['sessions_run']} sessions so far.", flush=True)

    print("[ablation] Building one index per ablation config (once per run, not per session)...", flush=True)
    indexes = {
        label: _build_ablation_index(subset, keep, num_threads=16)
        for label, keep in configs.items()
    }
    for label, vf in indexes.items():
        print(f"  - {label}: dim={vf.vectors.shape[1]}", flush=True)

    print("[ablation] Starting sessions...", flush=True)
    session = 0
    try:
        while max_sessions is None or session < max_sessions:
            if _stop_requested:
                break

            session_seed = state["seed_base"] + state["sessions_run"]
            per_combo = max(1, cases_per_edit_level)
            cases = generate_error_cases(subset, per_combo, DEFAULT_EDIT_LEVELS, session_seed)
            queries = [str(c["query"]) for c in cases]

            for label, vf in indexes.items():
                preds = lookup_vecfuzz(vf, queries, k)
                accumulate_accuracy(state["ablations"][label], cases, preds)

            state["sessions_run"] += 1
            session += 1
            save_json(path, state)

            print(
                "\r"
                f"[ablation] session {state['sessions_run']} done ",
                end="",
                flush=True,
            )
    finally:
        pass

    print(f"\n[ablation] Stopped after {state['sessions_run']} total sessions. State saved to {path}.")
    return state


def main() -> None:
    p = argparse.ArgumentParser(description="Resumable ablation benchmark for VecFuzz sub-vectors.")
    p.add_argument("--vocab-size", type=int, default=150_000)
    p.add_argument("--cases", type=int, default=1_000)
    p.add_argument("--max-sessions", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--max-words", type=int, default=None)
    p.add_argument("--plot", action="store_true", help="Render the per-tier figures from existing state and exit.")
    p.add_argument("--output-dir", default="benchmark_outputs")
    args = p.parse_args()

    if args.plot:
        state = load_json(Path(f"{args.output_dir}/ablation_state_{args.vocab_size}_k{args.k}.json"))
        if state is None:
            raise SystemExit(f"No state file at {args.output_dir} to plot.")
        plot_all_groups(state, args.k, args.output_dir)
        return

    run_ablation_benchmark(
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