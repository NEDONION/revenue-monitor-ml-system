from pathlib import Path
from typing import List

import pandas as pd


def _read_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_processed_dataset(input_dir: Path, split: str) -> pd.DataFrame:
    # 优先读取按分片输出的文件，找不到则合并目录下全部数据。
    candidates: List[Path] = list(input_dir.glob(f"*_{split}.parquet"))
    if not candidates:
        candidates = list(input_dir.glob(f"*_{split}.csv"))
    if not candidates:
        candidates = list(input_dir.glob("*.parquet")) + list(input_dir.glob("*.csv"))

    if not candidates:
        raise FileNotFoundError(f"No dataset files found under {input_dir}")

    frames = [_read_file(path) for path in candidates]
    return pd.concat(frames, ignore_index=True)


def filter_granularity(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if "granularity" not in df.columns:
        return df
    return df[df["granularity"] == granularity].copy()
