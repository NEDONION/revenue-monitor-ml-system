import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.training.data import filter_granularity, load_processed_dataset
from src.training.tcn_model import TCN
from src.training.torch_data import SequenceConfig, SequenceDataset


@dataclass(frozen=True)
class TCNConfig:
    input_dir: Path
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
    epochs: int
    learning_rate: float


def load_tcn_config(path: Path) -> TCNConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return TCNConfig(
        input_dir=Path(raw["input_dir"]),
        output_dir=Path(raw["output_dir"]),
        split=raw.get("split", "train"),
        granularity=raw.get("granularity", "hourly"),
        series_id_column=raw.get("series_id_column", "series_id"),
        time_column=raw.get("time_column", "ts"),
        value_column=raw.get("value_column", "metric_value"),
        context_columns=raw.get("context_columns", []),
        input_length=int(raw.get("input_length", 168)),
        horizon=int(raw.get("horizon", 24)),
        batch_size=int(raw.get("batch_size", 128)),
        epochs=int(raw.get("epochs", 5)),
        learning_rate=float(raw.get("learning_rate", 1e-3)),
    )


def ensure_context_columns(df: pd.DataFrame, context_columns: List[str]) -> List[str]:
    # 只保留存在的上下文字段，避免训练时报错。
    return [col for col in context_columns if col in df.columns]


def train_tcn(config_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = load_tcn_config(config_path)
    logging.info("读取配置：%s", config_path)
    df = load_processed_dataset(config.input_dir, config.split)
    df = filter_granularity(df, config.granularity)
    logging.info("加载数据完成，行数：%d", len(df))

    context_cols = ensure_context_columns(df, config.context_columns)
    if not context_cols:
        logging.warning("未找到上下文字段，仅使用数值序列训练。")

    seq_config = SequenceConfig(
        input_length=config.input_length,
        horizon=config.horizon,
        value_column=config.value_column,
        context_columns=context_cols,
        time_column=config.time_column,
    )
    dataset = SequenceDataset(df, config.series_id_column, seq_config)
    if len(dataset) == 0:
        raise ValueError("可用训练样本为 0，请检查输入长度与窗口配置。")
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    input_channels = 1 + len(context_cols)
    model = TCN(input_channels=input_channels, hidden_channels=32, num_layers=3, horizon=config.horizon)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.MSELoss()

    logging.info("开始训练：epochs=%d, batch=%d", config.epochs, config.batch_size)
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataset)
        logging.info("Epoch %d/%d - loss: %.6f", epoch + 1, config.epochs, avg_loss)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.output_dir / "tcn_model.pt"
    torch.save(model.state_dict(), model_path)
    logging.info("模型保存完成：%s", model_path)


__all__ = ["train_tcn", "load_tcn_config", "TCNConfig"]
