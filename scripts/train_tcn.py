#!/usr/bin/env python3
from pathlib import Path

from src.training.tcn_train import train_tcn


def main() -> None:
    train_tcn(Path("configs/tcn_training.json"))


if __name__ == "__main__":
    main()
