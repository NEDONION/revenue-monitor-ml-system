import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch

from src.training.data import filter_granularity, load_processed_dataset
from src.training.tcn import TCN

MIN_STD = 1e-3

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
    target_step: int
    batch_size: int
    single_result: bool
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
        target_step=int(raw.get("target_step", 1)),
        batch_size=int(raw.get("batch_size", 256)),
        single_result=bool(raw.get("single_result", True)),
        filters=raw.get("filters", {}),
    )


def load_infer_config_from_dict(raw: dict) -> TCNInferConfig:
    # 从字典构建推理配置（用于 API 请求）。
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
        target_step=int(raw.get("target_step", 1)),
        batch_size=int(raw.get("batch_size", 256)),
        single_result=bool(raw.get("single_result", True)),
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
) -> Tuple[torch.Tensor, List[Tuple[str, pd.Timestamp]], List[float]]:
    # 为每条序列取最新的输入窗口。
    batch_inputs: List[torch.Tensor] = []
    meta: List[Tuple[str, pd.Timestamp]] = []
    window_stds: List[float] = []
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
        window_stds.append(float(values.std()))

    if not batch_inputs:
        lengths = df.groupby(series_id_column).size()
        min_len = int(lengths.min()) if not lengths.empty else 0
        raise ValueError(
            f"可用于推理的序列为空（最短序列长度={min_len}，输入窗口={input_length}）。"
            "请检查输入数据长度或窗口配置。"
        )

    return torch.stack(batch_inputs, dim=0), meta, window_stds


def infer_single(config: TCNInferConfig) -> dict:
    # 执行一次推理，返回单条结果字典。
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
    applied_filters = []
    logging.info("推理过滤条件：%s", filters)
    for key in ("site", "currency", "fee_type", "series_id"):
        value = filters.get(key)
        if not value:
            continue
        if key not in df.columns:
            logging.warning("过滤字段不存在：%s", key)
            continue
        if key in ("site", "currency"):
            df = df[df[key].astype(str).str.upper() == str(value).upper()]
        elif key == "fee_type":
            df = df[df[key].astype(str).str.lower() == str(value).lower()]
        else:
            df = df[df[key].astype(str) == str(value)]
        applied_filters.append(f"{key}={value}")
        logging.info("应用过滤 %s=%s 后行数：%d", key, value, len(df))
    if applied_filters and df.empty:
        raise ValueError(f"过滤条件无匹配：{', '.join(applied_filters)}")
    logging.info("推理数据加载完成（含历史上下文），行数：%d", len(df))

    context_cols = _ensure_context_columns(df, config.context_columns)
    inputs, batch_meta, batch_stds = _build_infer_batches(
        df,
        config.series_id_column,
        config.time_column,
        config.value_column,
        context_cols,
        config.input_length,
    )

    input_channels = inputs.shape[-1]
    meta_path = config.model_dir / "tcn_meta.json"
    quantiles = [0.1, 0.9]
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            model_meta = json.load(handle)
        quantiles = model_meta.get("quantiles", quantiles)
        config = TCNInferConfig(
            input_dir=config.input_dir,
            model_dir=config.model_dir,
            output_path=config.output_path,
            granularity=config.granularity,
            series_id_column=config.series_id_column,
            time_column=config.time_column,
            value_column=config.value_column,
            context_columns=model_meta.get("context_columns", config.context_columns),
            input_length=int(model_meta.get("input_length", config.input_length)),
            horizon=int(model_meta.get("horizon", config.horizon)),
            target_step=min(int(config.target_step), int(model_meta.get("horizon", config.horizon))),
            batch_size=config.batch_size,
            single_result=config.single_result,
            filters=config.filters,
        )
    model = TCN(
        input_channels=input_channels,
        hidden_channels=32,
        num_layers=3,
        horizon=config.horizon,
        num_quantiles=len(quantiles),
    )
    model_path = config.model_dir / "tcn_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        preds = model(inputs)

    target_idx = max(1, config.target_step)
    (series_id, last_ts), row = batch_meta[0], preds[0]
    if len(quantiles) >= 2:
        lower = float(row[target_idx - 1, 0])
        upper = float(row[target_idx - 1, -1])
        if len(quantiles) >= 3:
            mid_idx = len(quantiles) // 2
            value = float(row[target_idx - 1, mid_idx])
        else:
            value = float(row[target_idx - 1, 0])
    else:
        value = float(row[target_idx - 1, 0])
        std = max(batch_stds[0] if batch_stds else 0.0, MIN_STD)
        lower = float(value) - 2 * std
        upper = float(value) + 2 * std
    return {
        "series_id": str(series_id),
        "last_ts": last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts),
        "horizon_step": int(target_idx),
        "prediction": float(value),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
    }


def run_tcn_inference(config_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = load_infer_config(config_path)
    logging.info("读取推理配置：%s", config_path)
    record = infer_single(config)
    output = pd.DataFrame([record])
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(config.output_path, index=False)
    json_path = config.output_path.with_suffix(".json")
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("推理完成，输出：%s", config.output_path)


__all__ = ["run_tcn_inference", "load_infer_config", "load_infer_config_from_dict", "infer_single", "TCNInferConfig"]
