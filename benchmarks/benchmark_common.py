from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence
from unicodedata import normalize
from pympler import asizeof
from spellchecker import SpellChecker
from symspellpy import SymSpell, Verbosity


from vecfuzz import VecFuzz

TYPO_TYPES = ("substitution", "transposition", "deletion", "insertion")
DEFAULT_EDIT_LEVELS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
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


def safe_size_mb(obj: object) -> float:
    try:
        return asizeof.asizeof(obj) / (1024 * 1024)
    except Exception:
        return 0.0


def normalize_word(word: str) -> str:
    return normalize("NFD", word).lower().strip()


def load_vocabulary(min_length: int = 4, max_words: Optional[int] = None, seed: int = 0) -> List[str]:
    freq_dict = SpellChecker().word_frequency.dictionary
    vocab = [normalize_word(w) for w in freq_dict.keys() if len(w) >= min_length]
    vocab = sorted({w for w in vocab if w})
    rng = random.Random(seed)
    rng.shuffle(vocab)
    if max_words is not None:
        vocab = vocab[:max_words]
    return vocab


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
    vocab: Sequence[str], cases_per_combo: int, edit_levels: Sequence[int], seed: int
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    cases: List[Dict[str, object]] = []
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


def build_symspell(vocab: Sequence[str], config: SymSpellConfig, frequencies: Dict[str, int]):
    t0 = perf_counter()
    instance = SymSpell(max_dictionary_edit_distance=config.max_edit_distance, prefix_length=config.prefix_length)
    for word in vocab:
        instance.create_dictionary_entry(word, max(1, frequencies.get(word, 1)))
    return instance, perf_counter() - t0


def build_vecfuzz(vocab: Sequence[str], num_threads: int | None = None):
    t0 = perf_counter()
    vf = VecFuzz(num_threads=num_threads).build(list(vocab))
    return vf, perf_counter() - t0

def lookup_vecfuzz(index, queries, k: int = 1):
    results = index.lookup(list(queries), k)
    return [[w for w, _d in c] for _q, c in results]


def lookup_symspell(index: SymSpell, queries, max_edit_distance: int, k: int = 1):
    out = []
    for q in queries:
        sug = index.lookup(q, Verbosity.CLOSEST, max_edit_distance=max_edit_distance)
        out.append([s.term for s in sug[:k]])
    return out


def empty_accuracy_accumulator() -> Dict[str, object]:
    return {
        "overall": {"count": 0.0, "recall1": 0.0},
        "by_error_type": {t: {"count": 0.0, "recall1": 0.0} for t in TYPO_TYPES},
        "by_edits": {str(e): {"count": 0.0, "recall1": 0.0} for e in DEFAULT_EDIT_LEVELS},
        "by_error_and_edits": {
            t: {str(e): {"count": 0.0, "recall1": 0.0} for e in DEFAULT_EDIT_LEVELS} for t in TYPO_TYPES
        },
    }


def accumulate_accuracy(accumulator: Dict[str, object], test_cases, predictions) -> None:
    """Adds new session's raw hit/count data into an existing accumulator, in place."""
    for case, candidates in zip(test_cases, predictions):
        target = str(case["target"])
        error_type = str(case["error_type"])
        edits = str(case["edits"])
        hit = 1 if target in candidates else 0

        accumulator["overall"]["count"] += 1
        accumulator["overall"]["recall1"] += hit
        accumulator["by_error_type"][error_type]["count"] += 1
        accumulator["by_error_type"][error_type]["recall1"] += hit
        accumulator["by_edits"][edits]["count"] += 1
        accumulator["by_edits"][edits]["recall1"] += hit
        accumulator["by_error_and_edits"][error_type][edits]["count"] += 1
        accumulator["by_error_and_edits"][error_type][edits]["recall1"] += hit


def save_json(path: Path, data: Dict[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)  # atomic-ish swap, avoids corrupting the file if interrupted mid-write


def load_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)