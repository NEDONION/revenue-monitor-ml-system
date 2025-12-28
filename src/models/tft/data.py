from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

from src.training.data import load_processed_dataset, filter_granularity


def prepare_tft_dataframe(
    input_dir: Path,
    split: str,
    time_column: str,
    granularity: str,
    base_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    # 读取处理后的数据并创建 time_idx（以小时为单位）。
    df = load_processed_dataset(input_dir, split)
    df = filter_granularity(df, granularity)
    df[time_column] = pd.to_datetime(df[time_column], utc=True)
    base = base_ts if base_ts is not None else df[time_column].min()
    df["time_idx"] = ((df[time_column] - base).dt.total_seconds() / 3600).astype(int)
    return df


def add_time_index(
    df: pd.DataFrame, time_column: str, base_ts: pd.Timestamp
) -> pd.DataFrame:
    # 根据统一基准时间构造 time_idx，保证不同 split 对齐。
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column], utc=True)
    df["time_idx"] = ((df[time_column] - base_ts).dt.total_seconds() / 3600).astype(int)
    return df


def ensure_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    # 只保留存在的列。
    return [col for col in columns if col in df.columns]


def apply_filters(df: pd.DataFrame, filters: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
    # 应用推理过滤条件并返回实际生效的条件列表。
    applied: List[str] = []
    for key in ("site", "currency", "fee_type", "series_id"):
        value = filters.get(key)
        if not value:
            continue
        if key not in df.columns:
            continue
        if key in ("site", "currency"):
            df = df[df[key].astype(str).str.upper() == str(value).upper()]
        elif key == "fee_type":
            df = df[df[key].astype(str).str.lower() == str(value).lower()]
        else:
            df = df[df[key].astype(str) == str(value)]
        applied.append(f"{key}={value}")
    return df, applied
