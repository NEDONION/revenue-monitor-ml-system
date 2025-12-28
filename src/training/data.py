from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _read_file(path: Path) -> pd.DataFrame:
    # 读取 CSV 或 Parquet。
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
    # 仅保留指定粒度的数据。
    if "granularity" not in df.columns:
        return df
    return df[df["granularity"] == granularity].copy()


@dataclass(frozen=True)
class SequenceConfig:
    # 序列样本的窗口配置。
    input_length: int
    horizon: int
    value_column: str
    context_columns: List[str]
    time_column: str


class SequenceDataset(Dataset):
    # 将长表数据切片为可训练的序列样本。
    def __init__(
        self,
        df: pd.DataFrame,
        series_id_column: str,
        config: SequenceConfig,
    ) -> None:
        self.df = df
        self.series_id_column = series_id_column
        self.config = config
        self.index = self._build_index()

    def _build_index(self) -> List[Tuple[int, int]]:
        # 为每条序列构建可用的 (start, end) 索引。
        index: List[Tuple[int, int]] = []
        for _, group in self.df.groupby(self.series_id_column):
            group = group.sort_values(self.config.time_column)
            n = len(group)
            window = self.config.input_length + self.config.horizon
            if n < window:
                continue
            for start in range(0, n - window + 1):
                end = start + window
                index.append((group.index[start], group.index[end - 1]))
        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        start_idx, end_idx = self.index[idx]
        # 取出连续片段，保证按时间顺序。
        segment = self.df.loc[start_idx:end_idx].copy()
        segment = segment.sort_values(self.config.time_column)
        values = segment[self.config.value_column].to_numpy(dtype=np.float32)
        context = segment[self.config.context_columns].to_numpy(dtype=np.float32)
        x_len = self.config.input_length
        y_len = self.config.horizon
        x_value = values[:x_len]
        x_context = context[:x_len]
        y_value = values[x_len : x_len + y_len]

        # 拼接 value 与上下文特征作为输入。
        x = np.concatenate([x_value.reshape(-1, 1), x_context], axis=1)
        return torch.from_numpy(x), torch.from_numpy(y_value)
