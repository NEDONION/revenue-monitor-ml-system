import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch

from src.training.data import filter_granularity, load_processed_dataset
from src.training.tcn import TCN


@dataclass(frozen=True)
class TCNInferConfig:
    input_dir: Path
    model_dir: Path
    output_path: Path
    granularity: str
    series_id_column: str
    time_column: str
    value_column: str
    context_columns: List[str]
    input_length: int
    horizon: int
    steps: int
    batch_size: int
    filters: dict


def load_infer_config(path: Path) -> TCNInferConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return TCNInferConfig(
        input_dir=Path(raw["input_dir"]),
        model_dir=Path(raw["model_dir"]),
        output_path=Path(raw["output_path"]),
        granularity=raw.get("granularity", "hourly"),
        series_id_column=raw.get("series_id_column", "series_id"),
        time_column=raw.get("time_column", "ts"),
        value_column=raw.get("value_column", "metric_value"),
        context_columns=raw.get("context_columns", []),
        input_length=int(raw.get("input_length", 168)),
        horizon=int(raw.get("horizon", 24)),
        steps=int(raw.get("steps", raw.get("horizon", 24))),
        batch_size=int(raw.get("batch_size", 256)),
        filters=raw.get("filters", {}),
    )


def _ensure_context_columns(df: pd.DataFrame, context_columns: List[str]) -> List[str]:
    # 只保留存在的上下文字段。
    return [col for col in context_columns if col in df.columns]


def _build_infer_batches(
    df: pd.DataFrame,
    series_id_column: str,
    time_column: str,
    value_column: str,
    context_columns: List[str],
    input_length: int,
) -> Tuple[torch.Tensor, List[Tuple[str, pd.Timestamp]]]:
    # 为每条序列取最新的输入窗口。
    batch_inputs: List[torch.Tensor] = []
    meta: List[Tuple[str, pd.Timestamp]] = []
    for series_id, group in df.groupby(series_id_column):
        group = group.sort_values(time_column)
        if len(group) < input_length:
            continue
        window = group.tail(input_length)
        values = window[value_column].to_numpy(dtype="float32")
        context = window[context_columns].to_numpy(dtype="float32") if context_columns else None
        if context is not None and len(context_columns) > 0:
            features = torch.from_numpy(
                pd.concat(
                    [
                        pd.Series(values).to_frame(),
                        pd.DataFrame(context, columns=context_columns),
                    ],
                    axis=1,
                ).to_numpy(dtype="float32")
            )
        else:
            features = torch.from_numpy(values.reshape(-1, 1))
        batch_inputs.append(features)
        meta.append((series_id, window[time_column].iloc[-1]))

    if not batch_inputs:
        lengths = df.groupby(series_id_column).size()
        min_len = int(lengths.min()) if not lengths.empty else 0
        raise ValueError(
            f"可用于推理的序列为空（最短序列长度={min_len}，输入窗口={input_length}）。"
            "请检查输入数据长度或窗口配置。"
        )

    return torch.stack(batch_inputs, dim=0), meta


def run_tcn_inference(config_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = load_infer_config(config_path)
    logging.info("读取推理配置：%s", config_path)

    # 推理需要完整历史上下文，因此合并 train/val/test。
    frames = []
    for split in ("train", "val", "test"):
        try:
            frames.append(load_processed_dataset(config.input_dir, split))
        except FileNotFoundError:
            continue
    if not frames:
        raise FileNotFoundError(f"No dataset files found under {config.input_dir}")
    df = filter_granularity(pd.concat(frames, ignore_index=True), config.granularity)
    filters = config.filters or {}
    for key in ("site", "currency", "fee_type", "series_id"):
        value = filters.get(key)
        if value:
            df = df[df.get(key) == value]
    logging.info("推理数据加载完成（含历史上下文），行数：%d", len(df))

    context_cols = _ensure_context_columns(df, config.context_columns)
    inputs, meta = _build_infer_batches(
        df,
        config.series_id_column,
        config.time_column,
        config.value_column,
        context_cols,
        config.input_length,
    )

    input_channels = inputs.shape[-1]
    meta_path = config.model_dir / "tcn_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        config = TCNInferConfig(
            input_dir=config.input_dir,
            model_dir=config.model_dir,
            output_path=config.output_path,
            granularity=config.granularity,
            series_id_column=config.series_id_column,
            time_column=config.time_column,
            value_column=config.value_column,
            context_columns=meta.get("context_columns", config.context_columns),
            input_length=int(meta.get("input_length", config.input_length)),
            horizon=int(meta.get("horizon", config.horizon)),
            steps=min(int(config.steps), int(meta.get("horizon", config.horizon))),
            batch_size=config.batch_size,
            filters=config.filters,
        )
    model = TCN(input_channels=input_channels, hidden_channels=32, num_layers=3, horizon=config.horizon)
    model_path = config.model_dir / "tcn_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        preds = model(inputs)

    records = []
    for (series_id, last_ts), row in zip(meta, preds):
        for step, value in enumerate(row.tolist()[: config.steps], start=1):
            records.append(
                {
                    "series_id": series_id,
                    "last_ts": last_ts,
                    "horizon_step": step,
                    "prediction": value,
                }
            )

    output = pd.DataFrame(records)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(config.output_path, index=False)
    logging.info("推理完成，输出：%s", config.output_path)


__all__ = ["run_tcn_inference", "load_infer_config", "TCNInferConfig"]
