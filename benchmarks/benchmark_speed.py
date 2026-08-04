from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark_common import (
    DEFAULT_EDIT_LEVELS, DEFAULT_SYMSPELL_CONFIGS, SymSpellConfig,
    build_symspell, build_vecfuzz, generate_error_cases,
    load_vocabulary, lookup_symspell, lookup_vecfuzz, safe_size_mb,
)
from spellchecker import SpellChecker


def sweep(
    vocabulary: Sequence[str],
    frequencies: Dict[str, int],
    vocab_sizes: Sequence[int],
    configs: Sequence[SymSpellConfig],
    query_count: int,
    seed: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for offset, vocab_size in enumerate(vocab_sizes):
        print(f"\r[speed] Benchmarking vocab_size={vocab_size}...", end="", flush=True)

        subset = list(vocabulary[:vocab_size])

        vf_index, vf_build_s = build_vecfuzz(subset)
        vf_size = safe_size_mb(vf_index)

        symspell_pool = []
        for cfg in configs:
            index, build_s = build_symspell(subset, cfg, frequencies)
            symspell_pool.append({"config": cfg, "index": index, "build_seconds": build_s, "size_mb": safe_size_mb(index)})

        cases = generate_error_cases(
            subset,
            cases_per_combo=max(1, query_count // (len(DEFAULT_EDIT_LEVELS) * 4)),
            edit_levels=DEFAULT_EDIT_LEVELS,
            seed=seed + offset,
        )[:query_count]
        queries = [str(c["query"]) for c in cases]

        t0 = perf_counter()
        lookup_vecfuzz(vf_index, queries)
        vf_seconds = perf_counter() - t0
        vf_qps = len(queries) / vf_seconds if vf_seconds > 0 else 0.0

        symspell_results = []
        for entry in symspell_pool:
            t0 = perf_counter()
            lookup_symspell(entry["index"], queries, entry["config"].max_edit_distance)
            seconds = perf_counter() - t0
            symspell_results.append({
                "label": entry["config"].label,
                "build_seconds": entry["build_seconds"],
                "size_mb": entry["size_mb"],
                "qps": len(queries) / seconds if seconds > 0 else 0.0,
                "seconds": seconds,
            })

        rows.append({
            "vocab_size": vocab_size,
            "query_count": len(cases),
            "vecfuzz": {"build_seconds": vf_build_s, "size_mb": vf_size, "qps": vf_qps, "seconds": vf_seconds},
            "symspell": symspell_results,
        })

    print()
    return rows


def _figure_path(output_dir: Path, stem: str) -> Path:
    return output_dir / f"{stem}.png"


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


def plot_build_time(rows: Sequence[Dict[str, object]], output_dir: Path) -> Path:
    vocab_sizes = [r["vocab_size"] for r in rows]
    vec_build = [r["vecfuzz"]["build_seconds"] for r in rows]
    labels = [e["label"] for e in rows[0]["symspell"]] if rows else []
    builds = [[r["symspell"][i]["build_seconds"] for r in rows] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(12, 6))
    _plot_lines(ax, vocab_sizes, [("VecFuzz", vec_build)] + list(zip(labels, builds)),
                "Build time vs dictionary size (Lower is better)", "Dictionary size", "Build time (s)")
    fig.tight_layout()
    path = _figure_path(output_dir, "build_time_vs_vocab_size")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_memory_footprint(rows: Sequence[Dict[str, object]], output_dir: Path) -> Path:
    vocab_sizes = [r["vocab_size"] for r in rows]
    vec_size = [r["vecfuzz"]["size_mb"] for r in rows]
    labels = [e["label"] for e in rows[0]["symspell"]] if rows else []
    sizes = [[r["symspell"][i]["size_mb"] for r in rows] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(12, 6))
    _plot_lines(ax, vocab_sizes, [("VecFuzz", vec_size)] + list(zip(labels, sizes)),
                "Memory footprint vs dictionary size (Lower is better)", "Dictionary size", "Memory (MB)")
    fig.tight_layout()
    path = _figure_path(output_dir, "memory_footprint_vs_vocab_size")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_lookup_speed(rows: Sequence[Dict[str, object]], output_dir: Path) -> Path:
    vocab_sizes = [r["vocab_size"] for r in rows]
    vec_qps = [r["vecfuzz"]["qps"] for r in rows]
    labels = [e["label"] for e in rows[0]["symspell"]] if rows else []
    qps = [[r["symspell"][i]["qps"] for r in rows] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(12, 6))
    _plot_lines(ax, vocab_sizes, [("VecFuzz", vec_qps)] + list(zip(labels, qps)),
                "Lookup speed vs dictionary size (Higher is better)", "Dictionary size", "Queries / second")
    fig.tight_layout()
    path = _figure_path(output_dir, "lookup_speed_vs_vocab_size")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def run_speed_benchmark(
    output_dir: str = "benchmark_outputs",
    vocab_sizes: Sequence[int] = (5_000, 10_000, 20_000, 40_000, 60_000, 80_000, 100_000, 125_000, 150_000),
    query_count: int = 15_000,
    seed: int = 0,
    max_words: int = None,
    save_json_file: bool = True,
) -> Dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("[speed] Loading vocabulary...", flush=True)
    vocab = load_vocabulary(max_words=max_words, seed=seed)
    frequencies = SpellChecker().word_frequency.dictionary
    configs = [SymSpellConfig(**c) for c in DEFAULT_SYMSPELL_CONFIGS]
    vocab_sizes = [v for v in vocab_sizes if v <= len(vocab)] or [len(vocab)]

    print(f"[speed] Loaded {len(vocab)} words; sweeps={vocab_sizes}; query_count={query_count}", flush=True)
    rows = sweep(vocab, frequencies, vocab_sizes, configs, query_count, seed)

    print("[speed] Rendering figures...", flush=True)
    figures = {
        "build_time": str(plot_build_time(rows, output_path)),
        "memory_footprint": str(plot_memory_footprint(rows, output_path)),
        "lookup_speed": str(plot_lookup_speed(rows, output_path)),
    }

    results = {
        "metadata": {
            "seed": seed, "vocab_count": len(vocab), "vocab_sizes": list(vocab_sizes),
            "query_count": query_count, "configs": [c.__dict__ for c in configs],
        },
        "rows": rows,
        "figures": figures,
    }

    if save_json_file:
        result_path = output_path / "vecfuzz_symspell_speed_benchmark.json"
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        results["results_path"] = str(result_path)
        print(f"[speed] JSON written to {result_path}", flush=True)

    print("[speed] Done.", flush=True)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Build+lookup speed benchmark (VecFuzz vs SymSpell).")
    p.add_argument("--output-dir", default="benchmark_outputs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-words", type=int, default=None)
    p.add_argument("--vocab-sizes", type=int, nargs="+",
                    default=[5_000, 10_000, 20_000, 40_000, 60_000, 80_000, 100_000, 125_000, 150_000])
    p.add_argument("--query-count", type=int, default=100_000)
    p.add_argument("--no-json", action="store_true")
    args = p.parse_args()

    results = run_speed_benchmark(
        output_dir=args.output_dir,
        vocab_sizes=args.vocab_sizes,
        query_count=args.query_count,
        seed=args.seed,
        max_words=args.max_words,
        save_json_file=not args.no_json,
    )
    print(json.dumps(results["figures"], ensure_ascii=False, indent=2))
    if "results_path" in results:
        print(results["results_path"])


if __name__ == "__main__":
    main()