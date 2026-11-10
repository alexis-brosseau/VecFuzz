from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARKS_DIR.parent

BENCHMARKS = (
    "birkbeck.py",
    "benchmark_accuracy.py",
    "benchmark_speed.py",
)


def main() -> None:
    pythonpath = [str(PROJECT_ROOT)]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath.append(existing_pythonpath)

    env = {**os.environ, "PYTHONPATH": os.pathsep.join(pythonpath)}

    for benchmark in BENCHMARKS:
        print(f"[run] Starting {benchmark}...", flush=True)
        subprocess.run(
            [sys.executable, str(BENCHMARKS_DIR / benchmark)],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
        )

    print("[run] All benchmarks completed.", flush=True)


if __name__ == "__main__":
    main()
