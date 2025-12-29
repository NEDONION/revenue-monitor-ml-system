#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.tcn import train_tcn  # noqa: E402


def main() -> None:
    # 训练入口，默认读取 TCN 配置。
    train_tcn(Path("configs/tcn/train.json"))


if __name__ == "__main__":
    main()
