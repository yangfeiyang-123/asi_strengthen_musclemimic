"""Direct command-line entry point for reusable multi-session sEMG preprocessing."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from emg.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["preprocess-dataset", *sys.argv[1:]]))
