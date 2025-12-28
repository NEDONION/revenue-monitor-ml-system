#!/usr/bin/env python3
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.tcn_infer import run_tcn_inference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/tcn_infer.json"))
    args = parser.parse_args()
    # 推理入口，默认读取 TCN 推理配置。
    run_tcn_inference(args.config)


if __name__ == "__main__":
    main()
