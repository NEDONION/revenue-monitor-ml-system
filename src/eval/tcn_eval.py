import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.eval.metrics import eval_metrics
from src.training.data import filter_granularity, load_processed_dataset
from src.training.tcn import TCN


@dataclass(frozen=True)
class TCNEvalConfig:
    input_dir: Path
    model_dir: Path
    output_dir: Path
    split: str
    granularity: str
    series_id_column: str
    time_column: str
    value_column: str
    context_columns: List[str]
    input_length: int
    horizon: int
    batch_size: int
    hidden_channels: int
    num_layers: int


def load_eval_config(path: Path) -> TCNEvalConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return TCNEvalConfig(
        input_dir=Path(raw["input_dir"]),
        model_dir=Path(raw["model_dir"]),
        output_dir=Path(raw["output_dir"]),
        split=raw.get("split", "test"),
        granularity=raw.get("granularity", "hourly"),
        series_id_column=raw.get("series_id_column", "series_id"),
        time_column=raw.get("time_column", "ts"),
        value_column=raw.get("value_column", "metric_value"),
        context_columns=raw.get("context_columns", []),
        input_length=int(raw.get("input_length", 168)),
        horizon=int(raw.get("horizon", 24)),
        batch_size=int(raw.get("batch_size", 128)),
        hidden_channels=int(raw.get("hidden_channels", 32)),
        num_layers=int(raw.get("num_layers", 3)),
    )


def _ensure_context_columns(df: pd.DataFrame, context_columns: List[str]) -> List[str]:
    # 只保留存在的上下文字段。
    return [col for col in context_columns if col in df.columns]


def _load_all_splits(input_dir: Path) -> pd.DataFrame:
    # 合并 train/val/test 三个分片（存在即读取）。
    frames = []
    for split in ("train", "val", "test"):
        try:
            frames.append(load_processed_dataset(input_dir, split))
        except FileNotFoundError:
            continue
    if not frames:
        raise FileNotFoundError(f"No dataset files found under {input_dir}")
    return pd.concat(frames, ignore_index=True)


def _build_eval_tensors(
    df: pd.DataFrame,
    series_id_column: str,
    time_column: str,
    value_column: str,
    context_columns: List[str],
    input_length: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 用全量数据构造样本，但只评估 target 落在 test 分片的窗口。
    xs = []
    ys = []
    for _, group in df.groupby(series_id_column):
        group = group.sort_values(time_column).reset_index(drop=True)
        if len(group) < input_length + horizon:
            continue
        for start in range(0, len(group) - input_length - horizon + 1):
            x_window = group.iloc[start : start + input_length]
            y_window = group.iloc[start + input_length : start + input_length + horizon]
            if "split" in group.columns:
                if not (y_window["split"] == "test").all():
                    continue
            values = x_window[value_column].to_numpy(dtype="float32")
            if context_columns:
                context = x_window[context_columns].to_numpy(dtype="float32")
                features = pd.concat(
                    [
                        pd.Series(values).to_frame(),
                        pd.DataFrame(context, columns=context_columns),
                    ],
                    axis=1,
                ).to_numpy(dtype="float32")
            else:
                features = values.reshape(-1, 1)
            xs.append(torch.from_numpy(features))
            ys.append(torch.from_numpy(y_window[value_column].to_numpy(dtype="float32")))
    if not xs:
        raise ValueError("评估样本为 0，请检查窗口与数据长度。")
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def evaluate_tcn(config_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = load_eval_config(config_path)
    logging.info("读取评估配置：%s", config_path)

    df = _load_all_splits(config.input_dir)
    df = filter_granularity(df, config.granularity)
    logging.info("评估数据加载完成（含历史上下文），行数：%d", len(df))

    context_cols = _ensure_context_columns(df, config.context_columns)
    xs, ys = _build_eval_tensors(
        df,
        config.series_id_column,
        config.time_column,
        config.value_column,
        context_cols,
        config.input_length,
        config.horizon,
    )
    loader = DataLoader(list(zip(xs, ys)), batch_size=config.batch_size, shuffle=False)

    input_channels = 1 + len(context_cols)
    model = TCN(
        input_channels=input_channels,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        horizon=config.horizon,
    )
    model_path = config.model_dir / "tcn_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    preds_list = []
    targets_list = []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x)
            preds_list.append(preds)
            targets_list.append(y)

    all_preds = torch.cat(preds_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)
    metrics = eval_metrics(all_preds, all_targets)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.output_dir / "tcn_eval.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    logging.info("评估完成，指标已保存：%s", report_path)


__all__ = ["evaluate_tcn", "load_eval_config", "TCNEvalConfig"]
