import json
from datetime import datetime
import time
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from src.training.data import (
    SequenceConfig,
    SequenceDataset,
    filter_granularity,
    load_processed_dataset,
)

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None


@dataclass(frozen=True)
class TCNConfig:
    # 训练配置与模型超参数。
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
    hidden_channels: int
    num_layers: int
    loss_type: str
    quantiles: List[float]
    early_stopping_patience: int
    early_stopping_min_delta: float


def load_tcn_config(path: Path) -> TCNConfig:
    # 从 JSON 加载训练配置。
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
        hidden_channels=int(raw.get("hidden_channels", 32)),
        num_layers=int(raw.get("num_layers", 3)),
        loss_type=raw.get("loss_type", "mae"),
        quantiles=list(raw.get("quantiles", [0.1, 0.9])),
        early_stopping_patience=int(raw.get("early_stopping_patience", 3)),
        early_stopping_min_delta=float(raw.get("early_stopping_min_delta", 1e-4)),
    )


class TCN(nn.Module):
    # 时间卷积网络（TCN）用于多步预测。
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_layers: int,
        horizon: int,
        num_quantiles: int = 1,
    ) -> None:
        super().__init__()
        layers = []
        in_channels = input_channels
        for i in range(num_layers):
            dilation = 2**i
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                )
            )
            layers.append(nn.ReLU())
            in_channels = hidden_channels
        self.tcn = nn.Sequential(*layers)
        self.horizon = horizon
        self.num_quantiles = num_quantiles
        self.head = nn.Linear(hidden_channels, horizon * num_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.transpose(1, 2)
        features = self.tcn(x)
        # 取最后一个时间步作为序列表示。
        last = features[:, :, -1]
        out = self.head(last)
        return out.view(-1, self.horizon, self.num_quantiles)


def ensure_context_columns(df: pd.DataFrame, context_columns: List[str]) -> List[str]:
    # 只保留存在的上下文字段，避免训练时报错。
    return [col for col in context_columns if col in df.columns]


def _select_loss(loss_type: str, quantiles: List[float]):
    # 根据配置选择损失函数。
    if loss_type == "mae":
        return torch.nn.L1Loss()
    if loss_type == "mse":
        return torch.nn.MSELoss()
    if loss_type == "quantile":
        taus = torch.tensor(quantiles).view(1, 1, -1)
        def _pinball(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            # pred: (batch, horizon, q), target: (batch, horizon)
            diff = target.unsqueeze(-1) - pred
            loss = torch.maximum(taus * diff, (taus - 1) * diff)
            return torch.mean(loss)
        return _pinball
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def _eval_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    # 计算常用回归指标（MAE/RMSE/MAPE/SMAPE/R2）。
    abs_diff = torch.abs(preds - targets)
    mae = torch.mean(abs_diff).item()
    rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()

    eps = 1e-8
    mape = torch.mean(abs_diff / (torch.abs(targets) + eps)).item()
    smape = torch.mean(2 * abs_diff / (torch.abs(preds) + torch.abs(targets) + eps)).item()

    target_mean = torch.mean(targets)
    ss_tot = torch.sum((targets - target_mean) ** 2)
    ss_res = torch.sum((targets - preds) ** 2)
    r2 = (1 - ss_res / (ss_tot + eps)).item()

    p50 = torch.quantile(abs_diff, 0.5).item()
    p90 = torch.quantile(abs_diff, 0.9).item()

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "r2": r2,
        "p50_err": p50,
        "p90_err": p90,
    }


def train_tcn(config_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 加载 .env，便于读取 W&B 等配置。
    load_dotenv()

    config = load_tcn_config(config_path)
    logging.info("读取配置：%s", config_path)

    train_df = load_processed_dataset(config.input_dir, "train")
    train_df = filter_granularity(train_df, config.granularity)
    logging.info("训练数据加载完成，行数：%d", len(train_df))

    val_df = load_processed_dataset(config.input_dir, "val")
    val_df = filter_granularity(val_df, config.granularity)
    logging.info("验证数据加载完成，行数：%d", len(val_df))

    context_cols = ensure_context_columns(train_df, config.context_columns)
    if not context_cols:
        logging.warning("未找到上下文字段，仅使用数值序列训练。")

    seq_config = SequenceConfig(
        input_length=config.input_length,
        horizon=config.horizon,
        value_column=config.value_column,
        context_columns=context_cols,
        time_column=config.time_column,
    )
    train_dataset = SequenceDataset(train_df, config.series_id_column, seq_config)
    if len(train_dataset) == 0:
        lengths = train_df.groupby(config.series_id_column).size()
        min_len = int(lengths.min()) if not lengths.empty else 0
        window = config.input_length + config.horizon
        logging.error(
            "可用训练样本为 0（最短序列长度=%d，窗口=%d）。请重跑数据构造或调整窗口。",
            min_len,
            window,
        )
        raise ValueError("可用训练样本为 0，请检查输入长度与窗口配置。")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = SequenceDataset(val_df, config.series_id_column, seq_config)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    input_channels = 1 + len(context_cols)
    model = TCN(
        input_channels=input_channels,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        horizon=config.horizon,
        num_quantiles=len(config.quantiles) if config.loss_type == "quantile" else 1,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = _select_loss(config.loss_type, config.quantiles)

    use_wandb = bool(os.environ.get("WANDB_PROJECT")) and wandb is not None
    run = None
    if use_wandb:
        run_name = os.environ.get("WANDB_RUN_NAME") or datetime.now().strftime("run-%Y%m%d-%H%M%S")
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT"),
            entity=os.environ.get("WANDB_ENTITY"),
            name=run_name,
            mode=os.environ.get("WANDB_MODE", "online"),
            config=vars(config),
        )

    logging.info(
        "开始训练：epochs=%d, batch=%d, train_batches=%d, val_batches=%d",
        config.epochs,
        config.batch_size,
        len(train_loader),
        len(val_loader),
    )
    model.train()
    best_val = float("inf")
    best_state = None
    patience = 0
    for epoch in range(config.epochs):
        epoch_start = time.time()
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start_time is not None:
            start_time.record()
        total_loss = 0.0
        total_samples = 0
        log_interval = max(1, len(train_loader) // 5)
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            if config.loss_type != "quantile":
                preds = preds.squeeze(-1)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            if batch_idx % log_interval == 0 or batch_idx == len(train_loader):
                avg_loss = total_loss / max(1, total_samples)
                logging.info(
                    "Epoch %d/%d - train batch %d/%d - avg_loss: %.6f",
                    epoch + 1,
                    config.epochs,
                    batch_idx,
                    len(train_loader),
                    avg_loss,
                )
        avg_loss = total_loss / len(train_dataset)

        # 验证集评估。
        model.eval()
        val_loss = None
        val_metrics = None
        if len(val_dataset) > 0:
            logging.info("Epoch %d/%d - 开始验证", epoch + 1, config.epochs)
            val_total = 0.0
            preds_list = []
            targets_list = []
            with torch.no_grad():
                val_log_interval = max(1, len(val_loader) // 3)
                for batch_idx, (x, y) in enumerate(val_loader, start=1):
                    x = x.to(device)
                    y = y.to(device)
                    preds = model(x)
                    if config.loss_type != "quantile":
                        preds = preds.squeeze(-1)
                    loss = loss_fn(preds, y)
                    val_total += loss.item() * x.size(0)
                    preds_list.append(preds.cpu())
                    targets_list.append(y.cpu())
                    if batch_idx % val_log_interval == 0 or batch_idx == len(val_loader):
                        logging.info(
                            "Epoch %d/%d - val batch %d/%d",
                            epoch + 1,
                            config.epochs,
                            batch_idx,
                            len(val_loader),
                        )
            val_loss = val_total / len(val_dataset)
            all_preds = torch.cat(preds_list, dim=0)
            all_targets = torch.cat(targets_list, dim=0)
            val_metrics = _eval_metrics(all_preds, all_targets)
        model.train()

        step_time = None
        if end_time is not None:
            end_time.record()
            torch.cuda.synchronize()
            step_time = start_time.elapsed_time(end_time) / 1000.0
        throughput = None
        if step_time and step_time > 0:
            throughput = total_samples / step_time

        if val_loss is not None:
            logging.info(
                "Epoch %d/%d - train: %.6f - val: %.6f - val_mae: %.6f - val_rmse: %.6f",
                epoch + 1,
                config.epochs,
                avg_loss,
                val_loss,
                val_metrics["mae"] if val_metrics else 0.0,
                val_metrics["rmse"] if val_metrics else 0.0,
            )
        else:
            logging.info("Epoch %d/%d - train: %.6f", epoch + 1, config.epochs, avg_loss)
        logging.info(
            "Epoch %d/%d - 用时 %.1fs",
            epoch + 1,
            config.epochs,
            time.time() - epoch_start,
        )

        if use_wandb and run is not None:
            log_payload = {
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "learning_rate": config.learning_rate,
            }
            if val_metrics is not None:
                log_payload.update(
                    {
                        "val_mae": val_metrics["mae"],
                        "val_rmse": val_metrics["rmse"],
                        "val_mape": val_metrics["mape"],
                        "val_smape": val_metrics["smape"],
                        "val_r2": val_metrics["r2"],
                        "val_p50_err": val_metrics["p50_err"],
                        "val_p90_err": val_metrics["p90_err"],
                    }
                )
            if throughput is not None:
                log_payload["throughput_samples_per_sec"] = throughput
            if torch.cuda.is_available():
                log_payload["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1_048_576
                log_payload["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1_048_576
            wandb.log(log_payload, step=epoch + 1)

        if val_loss is not None:
            if val_loss + config.early_stopping_min_delta < best_val:
                best_val = val_loss
                best_state = model.state_dict()
                patience = 0
            else:
                patience += 1
                if patience >= config.early_stopping_patience:
                    logging.info("早停触发，最佳验证损失：%.6f", best_val)
                    break

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.output_dir / "tcn_model.pt"
    if best_state is not None:
        torch.save(best_state, model_path)
    else:
        torch.save(model.state_dict(), model_path)
    logging.info("模型保存完成：%s", model_path)
    meta_path = config.output_dir / "tcn_meta.json"
    meta = {
        "input_length": config.input_length,
        "horizon": config.horizon,
        "context_columns": config.context_columns,
        "hidden_channels": config.hidden_channels,
        "num_layers": config.num_layers,
        "quantiles": config.quantiles,
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    logging.info("模型元数据已保存：%s", meta_path)

    if use_wandb and run is not None:
        run.finish()


__all__ = ["train_tcn", "load_tcn_config", "TCNConfig", "TCN"]
