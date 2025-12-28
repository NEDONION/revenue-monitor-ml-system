#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.tft.eval import evaluate_tft  # noqa: E402


def main() -> None:
    evaluate_tft(Path("configs/tft/eval.json"))


if __name__ == "__main__":
    main()
