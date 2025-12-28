from dataclasses import dataclass
from pathlib import Path
from typing import List
import json


@dataclass(frozen=True)
class TrainingConfig:
    input_dir: Path
    output_dir: Path
    split: str
    value_column: str
    series_id_column: str
    groupby_columns: List[str]
    quantiles: List[float]
    min_points_per_group: int


def load_config(path: Path) -> TrainingConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return TrainingConfig(
        input_dir=Path(raw["input_dir"]),
        output_dir=Path(raw["output_dir"]),
        split=raw.get("split", "train"),
        value_column=raw.get("value_column", "metric_value"),
        series_id_column=raw.get("series_id_column", "series_id"),
        groupby_columns=raw.get("groupby_columns", ["series_id", "local_hour"]),
        quantiles=raw.get("quantiles", [0.1, 0.9]),
        min_points_per_group=raw.get("min_points_per_group", 50),
    )
