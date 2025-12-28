import json
import logging
from pathlib import Path

import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from src.eval.metrics import eval_metrics
from src.models.tft.config import load_eval_config
from src.models.tft.data import ensure_columns
from src.training.data import load_processed_dataset, filter_granularity


def evaluate_tft(config_path: Path) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_eval_config(config_path)
    logging.info("读取评估配置：%s", config_path)

    meta = {}
    meta_path = config.model_dir / "tft_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    time_column = meta.get("time_column", config.time_column)
    target_column = meta.get("target_column", config.target_column)
    group_id_column = meta.get("group_id_column", config.group_id_column)
    encoder_length = int(meta.get("encoder_length", config.encoder_length))
    prediction_length = int(meta.get("prediction_length", config.prediction_length))
    quantiles = list(meta.get("quantiles", config.quantiles))
    static_categoricals = meta.get("static_categoricals", config.static_categoricals)
    known_reals = meta.get("known_reals", config.known_reals)
    unknown_reals = meta.get("unknown_reals", config.unknown_reals)

    frames = []
    test_frame = None
    for split in ("train", "val", "test"):
        try:
            df = load_processed_dataset(config.input_dir, split)
        except FileNotFoundError:
            continue
        df = filter_granularity(df, config.granularity)
        frames.append(df)
        if split == "test":
            test_frame = df

    if not frames or test_frame is None or test_frame.empty:
        raise ValueError("评估数据为空，请检查处理后的数据。")

    df_all = pd.concat(frames, ignore_index=True)
    df_all[time_column] = pd.to_datetime(df_all[time_column], utc=True)
    base = df_all[time_column].min()
    df_all["time_idx"] = ((df_all[time_column] - base).dt.total_seconds() / 3600).astype(int)
    test_frame = test_frame.copy()
    test_frame[time_column] = pd.to_datetime(test_frame[time_column], utc=True)
    test_frame["time_idx"] = (
        (test_frame[time_column] - base).dt.total_seconds() / 3600
    ).astype(int)
    test_start = int(test_frame["time_idx"].min())

    static_categoricals = ensure_columns(df_all, static_categoricals)
    known_reals = ensure_columns(df_all, known_reals)
    unknown_reals = ensure_columns(df_all, unknown_reals)

    # 仅用于评估的数据集构建，限制预测从测试区间开始。
    dataset = TimeSeriesDataSet(
        df_all,
        time_idx="time_idx",
        target=target_column,
        group_ids=[group_id_column],
        max_encoder_length=encoder_length,
        max_prediction_length=prediction_length,
        static_categoricals=static_categoricals,
        time_varying_known_reals=["time_idx"] + known_reals,
        time_varying_unknown_reals=unknown_reals,
        min_prediction_idx=test_start,
    )
    dataloader = dataset.to_dataloader(train=False, batch_size=config.batch_size, num_workers=0)

    checkpoint = config.model_dir / "tft.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"模型文件不存在：{checkpoint}")
    tft = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint))
    tft.eval()
    with torch.no_grad():
        preds, x = tft.predict(dataloader, mode="quantiles", return_x=True)

    median_idx = min(len(quantiles) // 2, preds.shape[-1] - 1)
    pred_values = preds[..., median_idx]
    targets = x["decoder_target"]

    metrics = eval_metrics(pred_values.flatten(), targets.flatten())
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("评估完成，指标已保存：%s", config.output_path)
    return metrics


__all__ = ["evaluate_tft"]
