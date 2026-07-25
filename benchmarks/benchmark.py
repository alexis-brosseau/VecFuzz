from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pympler import asizeof
from spellchecker import SpellChecker
from symspellpy import SymSpell, Verbosity
from unicodedata import normalize

from vecfuzz import VecFuzz


TYPO_TYPES = ("substitution", "swap", "deletion", "insertion")
DEFAULT_EDIT_LEVELS = (1, 2, 3, 4, 5 , 6, 7, 8, 9)
DEFAULT_SYMSPELL_CONFIGS = (
	{"label": "SymSpell d2/p7", "max_edit_distance": 2, "prefix_length": 7},
	{"label": "SymSpell d3/p9", "max_edit_distance": 3, "prefix_length": 9},
	{"label": "SymSpell d4/p12", "max_edit_distance": 4, "prefix_length": 12},
)


@dataclass(frozen=True)
class SymSpellConfig:
	label: str
	max_edit_distance: int
	prefix_length: int


@dataclass
class IndexArtifact:
	name: str
	config_label: str
	index: object
	build_seconds: float
	size_mb: float


def _safe_size_mb(obj: object) -> float:
	try:
		return asizeof.asizeof(obj) / (1024 * 1024)
	except Exception:
		return 0.0


def _mean(values: Sequence[float]) -> float:
	if not values:
		return 0.0
	return sum(values) / len(values)


def _normalize_word(word: str) -> str:
	return normalize("NFD", word).lower().strip()


def load_vocabulary(min_length: int = 4, max_words: Optional[int] = None, seed: int = 0) -> List[str]:
	freq_dict = SpellChecker().word_frequency.dictionary
	vocab = [
		_normalize_word(word)
		for word in freq_dict.keys()
		if len(word) >= min_length
	]
	vocab = sorted({word for word in vocab if word})
	rng = random.Random(seed)
	rng.shuffle(vocab)
	if max_words is not None:
		vocab = vocab[:max_words]
	return vocab


def _apply_single_typo(word: str, typo_type: str, rng: random.Random) -> str:
	if not word:
		return word

	if typo_type == "substitution":
		if len(word) < 2:
			return word
		idx = rng.randrange(len(word))
		alphabet = "abcdefghijklmnopqrstuvwxyz"
		replacement_choices = alphabet.replace(word[idx], "")
		replacement = rng.choice(replacement_choices or alphabet)
		return word[:idx] + replacement + word[idx + 1 :]

	if typo_type == "swap":
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
		return word[:idx] + word[idx + 1 :]

	if typo_type == "insertion":
		idx = rng.randrange(len(word) + 1)
		letter = rng.choice("abcdefghijklmnopqrstuvwxyz")
		return word[:idx] + letter + word[idx:]

	raise ValueError(f"Unknown typo type: {typo_type}")


def generate_error_cases(
	vocab: Sequence[str],
	cases_per_combo: int,
	edit_levels: Sequence[int],
	seed: int,
) -> List[Dict[str, object]]:
	rng = random.Random(seed)
	cases: List[Dict[str, object]] = []

	for typo_type in TYPO_TYPES:
		for edits in edit_levels:
			for _ in range(cases_per_combo):
				target = rng.choice(vocab)
				query = target
				for _ in range(edits):
					query = _apply_single_typo(query, typo_type, rng)

				cases.append(
					{
						"query": query,
						"target": target,
						"error_type": typo_type,
						"edits": edits,
					}
				)

	rng.shuffle(cases)
	return cases


def build_symspell(vocab: Sequence[str], config: SymSpellConfig, frequencies: Dict[str, int]) -> IndexArtifact:
	t0 = perf_counter()
	instance = SymSpell(
		max_dictionary_edit_distance=config.max_edit_distance,
		prefix_length=config.prefix_length,
	)
	for word in vocab:
		instance.create_dictionary_entry(word, max(1, frequencies.get(word, 1)))
	build_seconds = perf_counter() - t0
	return IndexArtifact(
		name="SymSpell",
		config_label=config.label,
		index=instance,
		build_seconds=build_seconds,
		size_mb=_safe_size_mb(instance),
	)


def build_vecfuzz(vocab: Sequence[str]) -> IndexArtifact:
	t0 = perf_counter()
	vecfuzz = VecFuzz()
	index = vecfuzz.build_index(list(vocab))
	build_seconds = perf_counter() - t0
	return IndexArtifact(
		name="VecFuzz",
		config_label="default",
		index=index,
		build_seconds=build_seconds,
		size_mb=_safe_size_mb(index),
	)


def lookup_vecfuzz(index: object, queries: Sequence[str], k: int = 5) -> List[List[str]]:
	results = index.lookup(list(queries), k)
	return [[word for word, _dist in candidates] for _query, candidates in results]


def lookup_symspell(index: SymSpell, queries: Sequence[str], max_edit_distance: int, k: int = 5) -> List[List[str]]:
	predictions: List[List[str]] = []
	for query in queries:
		suggestions = index.lookup(query, Verbosity.CLOSEST, max_edit_distance=max_edit_distance)
		predictions.append([suggestion.term for suggestion in suggestions[:k]])
	return predictions


def _accuracy_from_predictions(test_cases: Sequence[Dict[str, object]], predictions: Sequence[Sequence[str]]) -> Dict[str, Dict[str, float]]:
	summary: Dict[str, Dict[str, float]] = {
		"overall": {"count": 0.0, "recall1": 0.0},
		"by_error_type": {error_type: {"count": 0.0, "recall1": 0.0} for error_type in TYPO_TYPES},
		"by_edits": {str(edit): {"count": 0.0, "recall1": 0.0} for edit in DEFAULT_EDIT_LEVELS},
		"by_error_and_edits": {
			error_type: {str(edit): {"count": 0.0, "recall1": 0.0} for edit in DEFAULT_EDIT_LEVELS}
			for error_type in TYPO_TYPES
		},
	}

	for case, candidates in zip(test_cases, predictions):
		target = str(case["target"])
		error_type = str(case["error_type"])
		edits = str(case["edits"])
		hit = target in candidates[:1]

		summary["overall"]["count"] += 1
		summary["overall"]["recall1"] += 1 if hit else 0

		summary["by_error_type"][error_type]["count"] += 1
		summary["by_error_type"][error_type]["recall1"] += 1 if hit else 0

		summary["by_edits"][edits]["count"] += 1
		summary["by_edits"][edits]["recall1"] += 1 if hit else 0

		summary["by_error_and_edits"][error_type][edits]["count"] += 1
		summary["by_error_and_edits"][error_type][edits]["recall1"] += 1 if hit else 0

	return summary


def evaluate_lookup(
	test_cases: Sequence[Dict[str, object]],
	vecfuzz_index: object,
	symspell_index: SymSpell,
	symspell_config: SymSpellConfig,
	k: int = 5,
) -> Dict[str, object]:
	queries = [str(case["query"]) for case in test_cases]

	t0 = perf_counter()
	vecfuzz_predictions = lookup_vecfuzz(vecfuzz_index, queries, k=k)
	vecfuzz_seconds = perf_counter() - t0

	t0 = perf_counter()
	symspell_predictions = lookup_symspell(symspell_index, queries, symspell_config.max_edit_distance, k=k)
	symspell_seconds = perf_counter() - t0

	return {
		"vecfuzz": {
			"seconds": vecfuzz_seconds,
			"qps": len(test_cases) / vecfuzz_seconds if vecfuzz_seconds > 0 else 0.0,
			"accuracy": _accuracy_from_predictions(test_cases, vecfuzz_predictions),
		},
		"symspell": {
			"seconds": symspell_seconds,
			"qps": len(test_cases) / symspell_seconds if symspell_seconds > 0 else 0.0,
			"accuracy": _accuracy_from_predictions(test_cases, symspell_predictions),
		},
	}


def _build_symspell_pool(vocab: Sequence[str], frequencies: Dict[str, int], configs: Sequence[SymSpellConfig]) -> List[IndexArtifact]:
	return [build_symspell(vocab, config, frequencies) for config in configs]


def sweep_build_and_memory(
	vocabulary: Sequence[str],
	frequencies: Dict[str, int],
	vocab_sizes: Sequence[int],
	configs: Sequence[SymSpellConfig],
) -> List[Dict[str, object]]:
	rows: List[Dict[str, object]] = []
	for vocab_size in vocab_sizes:
		subset = list(vocabulary[:vocab_size])
		vecfuzz = build_vecfuzz(subset)
		symspell_pool = _build_symspell_pool(subset, frequencies, configs)

		rows.append(
			{
				"vocab_size": vocab_size,
				"vecfuzz": {"build_seconds": vecfuzz.build_seconds, "size_mb": vecfuzz.size_mb},
				"symspell": [
					{
						"label": artifact.config_label,
						"build_seconds": artifact.build_seconds,
						"size_mb": artifact.size_mb,
					}
					for artifact in symspell_pool
				],
			}
		)

	return rows


def sweep_lookup_by_vocab_size(
	vocabulary: Sequence[str],
	frequencies: Dict[str, int],
	vocab_sizes: Sequence[int],
	configs: Sequence[SymSpellConfig],
	queries_per_case: int,
	seed: int,
) -> List[Dict[str, object]]:
	rows: List[Dict[str, object]] = []
	for offset, vocab_size in enumerate(vocab_sizes):
		subset = list(vocabulary[:vocab_size])
		vecfuzz = build_vecfuzz(subset)
		symspell_pool = _build_symspell_pool(subset, frequencies, configs)

		cases = generate_error_cases(subset, cases_per_combo=max(1, queries_per_case // (len(TYPO_TYPES) * len(DEFAULT_EDIT_LEVELS))), edit_levels=DEFAULT_EDIT_LEVELS, seed=seed + offset)
		cases = cases[:queries_per_case]

		symspell_results = []
		for artifact, config in zip(symspell_pool, configs):
			lookup_results = evaluate_lookup(cases, vecfuzz.index, artifact.index, config)
			symspell_results.append(
				{
					"label": artifact.config_label,
					"qps": lookup_results["symspell"]["qps"],
					"seconds": lookup_results["symspell"]["seconds"],
				}
			)

		vecfuzz_lookup = evaluate_lookup(cases, vecfuzz.index, symspell_pool[0].index, configs[0])
		rows.append(
			{
				"vocab_size": vocab_size,
				"query_count": len(cases),
				"vecfuzz": {
					"qps": vecfuzz_lookup["vecfuzz"]["qps"],
					"seconds": vecfuzz_lookup["vecfuzz"]["seconds"],
				},
				"symspell": symspell_results,
			}
		)

	return rows


def sweep_accuracy(
	vocabulary: Sequence[str],
	frequencies: Dict[str, int],
	configs: Sequence[SymSpellConfig],
	cases_per_combo: int,
	seed: int,
) -> Dict[str, object]:
	vecfuzz = build_vecfuzz(vocabulary)
	symspell_pool = _build_symspell_pool(vocabulary, frequencies, configs)
	cases = generate_error_cases(vocabulary, cases_per_combo=cases_per_combo, edit_levels=DEFAULT_EDIT_LEVELS, seed=seed)

	accuracy_rows = {
		"vecfuzz": {},
		"symspell": [],
	}

	for artifact, config in zip(symspell_pool, configs):
		scores = evaluate_lookup(cases, vecfuzz.index, artifact.index, config)
		accuracy_rows["symspell"].append(
			{
				"label": artifact.config_label,
				"accuracy": scores["symspell"]["accuracy"],
			}
		)

	vec_scores = evaluate_lookup(cases, vecfuzz.index, symspell_pool[0].index, configs[0])
	accuracy_rows["vecfuzz"] = vec_scores["vecfuzz"]["accuracy"]

	return {
		"cases": cases,
		"vecfuzz": accuracy_rows["vecfuzz"],
		"symspell": accuracy_rows["symspell"],
	}


def _figure_path(output_dir: Path, stem: str) -> Path:
	return output_dir / f"{stem}.png"


def _plot_lines(ax, x_values: Sequence[int], series: Sequence[Tuple[str, Sequence[float]]], title: str, xlabel: str, ylabel: str) -> None:
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
			x_values,
			y_values,
			marker="o",
			linewidth=2.5 if is_vecfuzz else 2,
			label=label,
			color=color_map.get(label, palette[idx % len(palette)]),
			zorder=10 if is_vecfuzz else 2,
		)
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.grid(True, alpha=0.25)
	ax.legend(frameon=False)


def plot_build_time(results: Sequence[Dict[str, object]], output_dir: Path) -> Path:
	vocab_sizes = [row["vocab_size"] for row in results]
	vec_build = [row["vecfuzz"]["build_seconds"] for row in results]

	symspell_labels = [entry["label"] for entry in results[0]["symspell"]] if results else []
	symspell_builds = [[row["symspell"][idx]["build_seconds"] for row in results] for idx in range(len(symspell_labels))]

	fig, ax = plt.subplots(figsize=(12, 6))
	_plot_lines(
		ax,
		vocab_sizes,
		[("VecFuzz", vec_build)] + list(zip(symspell_labels, symspell_builds)),
		"Build time vs dictionary size (Lower is better)",
		"Dictionary size",
		"Build time (s)",
	)
	fig.tight_layout()
	path = _figure_path(output_dir, "build_time_vs_vocab_size")
	fig.savefig(path, dpi=180, bbox_inches="tight")
	plt.close(fig)
	return path


def plot_memory_footprint(results: Sequence[Dict[str, object]], output_dir: Path) -> Path:
	vocab_sizes = [row["vocab_size"] for row in results]
	vec_size = [row["vecfuzz"]["size_mb"] for row in results]

	symspell_labels = [entry["label"] for entry in results[0]["symspell"]] if results else []
	symspell_sizes = [[row["symspell"][idx]["size_mb"] for row in results] for idx in range(len(symspell_labels))]

	fig, ax = plt.subplots(figsize=(12, 6))
	_plot_lines(
		ax,
		vocab_sizes,
		[("VecFuzz", vec_size)] + list(zip(symspell_labels, symspell_sizes)),
		"Memory footprint vs dictionary size (Lower is better)",
		"Dictionary size",
		"Memory (MB)",
	)
	fig.tight_layout()
	path = _figure_path(output_dir, "memory_footprint_vs_vocab_size")
	fig.savefig(path, dpi=180, bbox_inches="tight")
	plt.close(fig)
	return path


def plot_lookup_by_vocab_size(results: Sequence[Dict[str, object]], output_dir: Path) -> Path:
	vocab_sizes = [row["vocab_size"] for row in results]
	vec_qps = [row["vecfuzz"]["qps"] for row in results]
	symspell_labels = [entry["label"] for entry in results[0]["symspell"]] if results else []
	symspell_qps = [[row["symspell"][idx]["qps"] for row in results] for idx in range(len(symspell_labels))]

	fig, ax = plt.subplots(figsize=(12, 6))
	_plot_lines(
		ax,
		vocab_sizes,
		[("VecFuzz", vec_qps)] + list(zip(symspell_labels, symspell_qps)),
		"Lookup speed vs dictionary size (Higher is better)",
		"Dictionary size",
		"Queries / second",
	)
	fig.tight_layout()
	path = _figure_path(output_dir, "lookup_speed_vs_vocab_size")
	fig.savefig(path, dpi=180, bbox_inches="tight")
	plt.close(fig)
	return path


def plot_accuracy(results: Dict[str, object], output_dir: Path) -> Path:
	vec_acc = results["vecfuzz"]
	symspell_results = results["symspell"]

	fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
	axes_flat = list(axes.flat)

	for idx, error_type in enumerate(TYPO_TYPES):
		ax = axes_flat[idx]
		x_values = list(DEFAULT_EDIT_LEVELS)
		# VecFuzz series.
		vec_series = []
		symspell_series = []

		vecfuzz_accuracy = vec_acc["by_error_and_edits"][error_type]
		vec_series = [
			("VecFuzz", [
				(vecfuzz_accuracy[str(edit)]["recall1"] / vecfuzz_accuracy[str(edit)]["count"]) if vecfuzz_accuracy[str(edit)]["count"] else 0.0
				for edit in DEFAULT_EDIT_LEVELS
			])
		]

		for symspell_result in symspell_results:
			label = symspell_result["label"]
			config_accuracy = symspell_result["accuracy"]["by_error_and_edits"][error_type]
			symspell_series.append(
				(
					label,
					[
						(config_accuracy[str(edit)]["recall1"] / config_accuracy[str(edit)]["count"]) if config_accuracy[str(edit)]["count"] else 0.0
						for edit in DEFAULT_EDIT_LEVELS
					],
				)
			)

		_plot_lines(
			ax,
			x_values,
			vec_series + symspell_series,
			f"{error_type.capitalize()} errors",
			"Number of edits",
			"Recall@1 accuracy",
		)
		ax.set_ylim(0, 1)

	fig.suptitle("Recall@1 accuracy by error type and number of edits (Higher is better)", y=1.02, fontsize=14)
	fig.tight_layout()
	path = _figure_path(output_dir, "accuracy_by_error_type_and_edits")
	fig.savefig(path, dpi=180, bbox_inches="tight")
	plt.close(fig)
	return path


def run_benchmark(
	output_dir: Union[str, Path] = "benchmark_outputs",
	vocab_sizes: Sequence[int] = (5_000, 10_000, 20_000, 40_000, 60_000, 80_000, 100_000),
	query_count: int = 100_000,
	seed: int = 0,
	max_words: Optional[int] = None,
	save_json: bool = True,
) -> Dict[str, object]:
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)
	print("[benchmark] Loading vocabulary and preparing inputs...", flush=True)

	vocab = load_vocabulary(max_words=max_words, seed=seed)
	frequencies = SpellChecker().word_frequency.dictionary
	configs = [SymSpellConfig(**config) for config in DEFAULT_SYMSPELL_CONFIGS]

	vocab_sizes = [size for size in vocab_sizes if size <= len(vocab)] or [len(vocab)]
	print(f"[benchmark] Loaded {len(vocab)} words; sweeps={vocab_sizes}; query_count={query_count}", flush=True)

	print("[benchmark] Running build/memory sweep...", flush=True)
	build_rows = sweep_build_and_memory(vocab, frequencies, vocab_sizes, configs)
	print("[benchmark] Running lookup speed by dictionary size...", flush=True)
	lookup_vocab_rows = sweep_lookup_by_vocab_size(vocab, frequencies, vocab_sizes, configs, queries_per_case=query_count, seed=seed + 10)
	print("[benchmark] Running accuracy sweep...", flush=True)
	accuracy_rows = sweep_accuracy(vocab[: max(vocab_sizes)], frequencies, configs, cases_per_combo=query_count, seed=seed + 30)

	print("[benchmark] Rendering figures...", flush=True)
	figures = {
		"build_time": str(plot_build_time(build_rows, output_path)),
		"memory_footprint": str(plot_memory_footprint(build_rows, output_path)),
		"lookup_vocab_size": str(plot_lookup_by_vocab_size(lookup_vocab_rows, output_path)),
		"accuracy": str(plot_accuracy(accuracy_rows, output_path)),
	}
	print("[benchmark] Figures rendered.", flush=True)

	results = {
		"metadata": {
			"seed": seed,
			"vocab_count": len(vocab),
			"vocab_sizes": list(vocab_sizes),
			"query_count": query_count,
			"configs": [asdict(config) for config in configs],
		},
		"build_and_memory": build_rows,
		"lookup_by_vocab_size": lookup_vocab_rows,
		"accuracy": accuracy_rows,
		"figures": figures,
	}

	if save_json:
		print("[benchmark] Writing JSON results...", flush=True)
		result_path = output_path / "vecfuzz_symspell_benchmark.json"
		with result_path.open("w", encoding="utf-8") as handle:
			json.dump(results, handle, ensure_ascii=False, indent=2)
		results["results_path"] = str(result_path)
		print(f"[benchmark] JSON written to {result_path}", flush=True)

	print("[benchmark] Done.", flush=True)

	return results


def main() -> None:
	parser = argparse.ArgumentParser(description="Benchmark VecFuzz vs SymSpell and generate Matplotlib figures.")
	parser.add_argument("--output-dir", default="benchmark_outputs", help="Directory where results and figures are written.")
	parser.add_argument("--seed", type=int, default=0, help="Random seed for vocabulary sampling and typo generation.")
	parser.add_argument("--max-words", type=int, default=None, help="Optional cap on the vocabulary size loaded from SpellChecker.")
	parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[5_000, 10_000, 20_000, 40_000, 60_000, 80_000, 100_000], help="Dictionary sizes to sweep.")
	parser.add_argument("--query-count", type=int, nargs="+", default=10_000, help="Query batch size to sweep.")
	parser.add_argument("--no-json", action="store_true", help="Do not write the raw JSON results file.")
	args = parser.parse_args()

	results = run_benchmark(
		output_dir=args.output_dir,
		vocab_sizes=args.vocab_sizes,
		query_count=args.query_count,
		seed=args.seed,
		max_words=args.max_words,
		save_json=not args.no_json,
	)

	print(json.dumps(results["figures"], ensure_ascii=False, indent=2))
	if "results_path" in results:
		print(results["results_path"])


if __name__ == "__main__":
	main()
