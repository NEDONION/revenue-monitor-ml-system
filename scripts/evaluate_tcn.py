#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.tcn_eval import evaluate_tcn  # noqa: E402


def main() -> None:
    # 评估入口，默认读取 TCN 评估配置。
    evaluate_tcn(Path("configs/tcn_eval.json"))


if __name__ == "__main__":
    main()
