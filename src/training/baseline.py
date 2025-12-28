from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass(frozen=True)
class QuantileModel:
    # 基线模型：保存分位数统计表。
    groupby_columns: List[str]
    quantiles: List[float]
    table: pd.DataFrame


def train_quantile_baseline(
    df: pd.DataFrame,
    groupby_columns: List[str],
    value_column: str,
    quantiles: List[float],
    min_points_per_group: int,
) -> QuantileModel:
    # 统计每个分组的分位数区间作为基线模型。
    grouped = df.groupby(groupby_columns)[value_column]
    counts = grouped.size().reset_index(name="n")
    q_table = grouped.quantile(quantiles).unstack(level=-1).reset_index()
    q_table = q_table.merge(counts, on=groupby_columns, how="left")
    q_table = q_table[q_table["n"] >= min_points_per_group].copy()

    # 规范列名，便于线上查询。
    q_table.columns = [
        *groupby_columns,
        *[f"q_{int(q * 100)}" for q in quantiles],
        "n",
    ]
    return QuantileModel(groupby_columns, quantiles, q_table)


def save_quantile_model(model: QuantileModel, output_dir: Path) -> None:
    # 输出分位数表，作为基线模型产物。
    output_dir.mkdir(parents=True, exist_ok=True)
    model.table.to_parquet(output_dir / "quantiles.parquet", index=False)
