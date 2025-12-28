from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import json


@dataclass(frozen=True)
class TFTTrainConfig:
    input_dir: Path
    output_dir: Path
    time_column: str
    target_column: str
    group_id_column: str
    static_categoricals: List[str]
    known_reals: List[str]
    unknown_reals: List[str]
    encoder_length: int
    prediction_length: int
    quantiles: List[float]
    max_epochs: int
    batch_size: int
    learning_rate: float
    granularity: str


@dataclass(frozen=True)
class TFTEvalConfig:
    input_dir: Path
    model_dir: Path
    output_path: Path
    time_column: str
    target_column: str
    group_id_column: str
    static_categoricals: List[str]
    known_reals: List[str]
    unknown_reals: List[str]
    encoder_length: int
    prediction_length: int
    quantiles: List[float]
    batch_size: int
    granularity: str


@dataclass(frozen=True)
class TFTInferConfig:
    input_dir: Path
    model_dir: Path
    output_path: Path
    time_column: str
    target_column: str
    group_id_column: str
    static_categoricals: List[str]
    known_reals: List[str]
    unknown_reals: List[str]
    encoder_length: int
    prediction_length: int
    quantiles: List[float]
    batch_size: int
    target_step: int
    granularity: str
    filters: Dict[str, str]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_train_config(path: Path) -> TFTTrainConfig:
    raw = _read_json(path)
    return TFTTrainConfig(
        input_dir=Path(raw["input_dir"]),
        output_dir=Path(raw["output_dir"]),
        time_column=raw.get("time_column", "ts"),
        target_column=raw.get("target_column", "metric_value"),
        group_id_column=raw.get("group_id_column", "series_id"),
        static_categoricals=list(raw.get("static_categoricals", [])),
        known_reals=list(raw.get("known_reals", [])),
        unknown_reals=list(raw.get("unknown_reals", [])),
        encoder_length=int(raw.get("encoder_length", 168)),
        prediction_length=int(raw.get("prediction_length", 24)),
        quantiles=list(raw.get("quantiles", [0.1, 0.5, 0.9])),
        max_epochs=int(raw.get("max_epochs", 10)),
        batch_size=int(raw.get("batch_size", 128)),
        learning_rate=float(raw.get("learning_rate", 1e-3)),
        granularity=raw.get("granularity", "hourly"),
    )


def load_eval_config(path: Path) -> TFTEvalConfig:
    raw = _read_json(path)
    return TFTEvalConfig(
        input_dir=Path(raw["input_dir"]),
        model_dir=Path(raw["model_dir"]),
        output_path=Path(raw["output_path"]),
        time_column=raw.get("time_column", "ts"),
        target_column=raw.get("target_column", "metric_value"),
        group_id_column=raw.get("group_id_column", "series_id"),
        static_categoricals=list(raw.get("static_categoricals", [])),
        known_reals=list(raw.get("known_reals", [])),
        unknown_reals=list(raw.get("unknown_reals", [])),
        encoder_length=int(raw.get("encoder_length", 168)),
        prediction_length=int(raw.get("prediction_length", 24)),
        quantiles=list(raw.get("quantiles", [0.1, 0.5, 0.9])),
        batch_size=int(raw.get("batch_size", 128)),
        granularity=raw.get("granularity", "hourly"),
    )


def load_infer_config(path: Path) -> TFTInferConfig:
    raw = _read_json(path)
    return TFTInferConfig(
        input_dir=Path(raw["input_dir"]),
        model_dir=Path(raw["model_dir"]),
        output_path=Path(raw["output_path"]),
        time_column=raw.get("time_column", "ts"),
        target_column=raw.get("target_column", "metric_value"),
        group_id_column=raw.get("group_id_column", "series_id"),
        static_categoricals=list(raw.get("static_categoricals", [])),
        known_reals=list(raw.get("known_reals", [])),
        unknown_reals=list(raw.get("unknown_reals", [])),
        encoder_length=int(raw.get("encoder_length", 168)),
        prediction_length=int(raw.get("prediction_length", 24)),
        quantiles=list(raw.get("quantiles", [0.1, 0.5, 0.9])),
        batch_size=int(raw.get("batch_size", 128)),
        target_step=int(raw.get("target_step", 1)),
        granularity=raw.get("granularity", "hourly"),
        filters=dict(raw.get("filters", {})),
    )


def load_infer_config_from_dict(raw: dict) -> TFTInferConfig:
    # 从字典构建推理配置（用于 API 请求）。
    return TFTInferConfig(
        input_dir=Path(raw["input_dir"]),
        model_dir=Path(raw["model_dir"]),
        output_path=Path(raw["output_path"]),
        time_column=raw.get("time_column", "ts"),
        target_column=raw.get("target_column", "metric_value"),
        group_id_column=raw.get("group_id_column", "series_id"),
        static_categoricals=list(raw.get("static_categoricals", [])),
        known_reals=list(raw.get("known_reals", [])),
        unknown_reals=list(raw.get("unknown_reals", [])),
        encoder_length=int(raw.get("encoder_length", 168)),
        prediction_length=int(raw.get("prediction_length", 24)),
        quantiles=list(raw.get("quantiles", [0.1, 0.5, 0.9])),
        batch_size=int(raw.get("batch_size", 128)),
        target_step=int(raw.get("target_step", 1)),
        granularity=raw.get("granularity", "hourly"),
        filters=dict(raw.get("filters", {})),
    )
