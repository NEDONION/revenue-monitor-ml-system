import json
import logging
from pathlib import Path

import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from src.models.tft.config import (
    TFTInferConfig,
    load_infer_config,
    load_infer_config_from_dict,
)
from src.models.tft.data import apply_filters, ensure_columns
from src.training.data import load_processed_dataset, filter_granularity


def _load_all_splits(config: TFTInferConfig, granularity: str, time_column: str) -> pd.DataFrame:
    # 汇总 train/val/test，供推理使用历史上下文。
    frames = []
    for split in ("train", "val", "test"):
        try:
            frames.append(load_processed_dataset(config.input_dir, split))
        except FileNotFoundError:
            continue
    if not frames:
        raise FileNotFoundError(f"未找到处理后的数据：{config.input_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = filter_granularity(df, granularity)
    df[time_column] = pd.to_datetime(df[time_column], utc=True)
    base = df[time_column].min()
    df["time_idx"] = ((df[time_column] - base).dt.total_seconds() / 3600).astype(int)
    return df


def infer_single(config: TFTInferConfig) -> dict:
    meta = {}
    meta_path = config.model_dir / "tft_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    granularity = meta.get("granularity", config.granularity)
    time_column = meta.get("time_column", config.time_column)
    target_column = meta.get("target_column", config.target_column)
    group_id_column = meta.get("group_id_column", config.group_id_column)
    encoder_length = int(meta.get("encoder_length", config.encoder_length))
    prediction_length = int(meta.get("prediction_length", config.prediction_length))
    quantiles = list(meta.get("quantiles", config.quantiles))
    static_categoricals = meta.get("static_categoricals", config.static_categoricals)
    known_reals = meta.get("known_reals", config.known_reals)
    unknown_reals = meta.get("unknown_reals", config.unknown_reals)

    df_all = _load_all_splits(config, granularity, time_column)
    df_filtered, applied = apply_filters(df_all, config.filters or {})
    if applied and df_filtered.empty:
        raise ValueError(f"过滤条件无匹配：{', '.join(applied)}")
    if df_filtered.empty:
        raise ValueError("推理数据为空，请检查过滤条件。")

    # 选取最近更新时间的序列作为推理目标。
    last_by_series = df_filtered.groupby(group_id_column)[time_column].max().sort_values()
    target_series = str(last_by_series.index[-1])
    df_series = df_filtered[df_filtered[group_id_column] == target_series]
    last_ts = df_series[time_column].max()
    logging.info("推理序列：%s", target_series)

    static_categoricals = ensure_columns(df_all, static_categoricals)
    known_reals = ensure_columns(df_all, known_reals)
    unknown_reals = ensure_columns(df_all, unknown_reals)

    # 使用全量数据构造训练数据集，确保类别编码一致。
    train_dataset = TimeSeriesDataSet(
        df_all,
        time_idx="time_idx",
        target=target_column,
        group_ids=[group_id_column],
        max_encoder_length=encoder_length,
        max_prediction_length=prediction_length,
        static_categoricals=static_categoricals,
        time_varying_known_reals=["time_idx"] + known_reals,
        time_varying_unknown_reals=unknown_reals,
    )
    predict_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        df_series,
        predict=True,
        stop_randomization=True,
    )
    dataloader = predict_dataset.to_dataloader(train=False, batch_size=config.batch_size, num_workers=0)

    checkpoint = config.model_dir / "tft.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"模型文件不存在：{checkpoint}")
    tft = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint))
    tft.eval()
    with torch.no_grad():
        preds = tft.predict(dataloader, mode="quantiles")

    target_step = max(1, min(config.target_step, preds.shape[1]))
    median_idx = min(len(quantiles) // 2, preds.shape[-1] - 1)
    lower = float(preds[0, target_step - 1, 0])
    upper = float(preds[0, target_step - 1, -1])
    prediction = float(preds[0, target_step - 1, median_idx])
    result = {
        "series_id": target_series,
        "last_ts": last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts),
        "horizon_step": int(target_step),
        "prediction": prediction,
        "lower_bound": lower,
        "upper_bound": upper,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("推理完成，输出：%s", config.output_path)
    return result


def infer_tft(config_path: Path) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_infer_config(config_path)
    logging.info("读取推理配置：%s", config_path)
    return infer_single(config)


def infer_tft_from_dict(payload: dict) -> dict:
    # 提供给 API 直接调用的入口。
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_infer_config_from_dict(payload)
    return infer_single(config)


__all__ = ["infer_tft", "infer_tft_from_dict", "infer_single"]
