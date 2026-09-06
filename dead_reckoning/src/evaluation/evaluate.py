"""Convenience evaluation wrapper redirecting to run_io_vnbd_benchmark."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.run_io_vnbd_benchmark import run_benchmark

if __name__ == "__main__":
    run_benchmark()
