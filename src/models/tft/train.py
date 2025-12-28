import json
import logging
import os
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
try:
    from lightning.pytorch.loggers import WandbLogger
except Exception:  # pragma: no cover - optional dependency
    WandbLogger = None
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from src.models.tft.config import load_train_config
from src.models.tft.data import ensure_columns, prepare_tft_dataframe


def train_tft(config_path: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_train_config(config_path)
    logging.info("读取训练配置：%s", config_path)
    df = prepare_tft_dataframe(
        config.input_dir,
        split="train",
        config.time_column,
        granularity=config.granularity,
    )
    static_categoricals = ensure_columns(df, config.static_categoricals)
    known_reals = ensure_columns(df, config.known_reals)
    unknown_reals = ensure_columns(df, config.unknown_reals)

    # 组装训练数据集，包含静态特征与时间序列特征。
    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=config.target_column,
        group_ids=[config.group_id_column],
        max_encoder_length=config.encoder_length,
        max_prediction_length=config.prediction_length,
        static_categoricals=static_categoricals,
        time_varying_known_reals=["time_idx"] + known_reals,
        time_varying_unknown_reals=unknown_reals,
    )
    dataloader = dataset.to_dataloader(train=True, batch_size=config.batch_size, num_workers=0)

    # 构建 TFT 模型并使用分位数损失。
    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=config.learning_rate,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        loss=QuantileLoss(quantiles=config.quantiles),
    )

    logger = False
    if os.environ.get("WANDB_PROJECT") and WandbLogger is not None:
        run_name = os.environ.get("WANDB_RUN_NAME") or datetime.now().strftime("run-%Y%m%d-%H%M%S")
        logger = WandbLogger(
            project=os.environ.get("WANDB_PROJECT"),
            entity=os.environ.get("WANDB_ENTITY"),
            name=run_name,
            mode=os.environ.get("WANDB_MODE", "online"),
        )
    trainer = pl.Trainer(max_epochs=config.max_epochs, enable_checkpointing=True, logger=logger)
    logging.info("开始训练：epochs=%d, batch=%d", config.max_epochs, config.batch_size)
    trainer.fit(tft, dataloader)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "tft.ckpt"
    trainer.save_checkpoint(str(checkpoint_path))

    meta = {
        "quantiles": config.quantiles,
        "encoder_length": config.encoder_length,
        "prediction_length": config.prediction_length,
        "time_column": config.time_column,
        "target_column": config.target_column,
        "group_id_column": config.group_id_column,
        "static_categoricals": static_categoricals,
        "known_reals": known_reals,
        "unknown_reals": unknown_reals,
        "granularity": config.granularity,
    }
    (config.output_dir / "tft_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info("模型保存完成：%s", checkpoint_path)
