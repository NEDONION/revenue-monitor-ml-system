#!/usr/bin/env python3
import logging
from pathlib import Path

from src.training.baseline import save_quantile_model, train_quantile_baseline
from src.training.config import load_config
from src.training.data import load_processed_dataset


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = load_config(Path("configs/training.json"))
    logging.info("读取配置：%s", Path("configs/training.json"))
    logging.info("加载数据：%s", config.input_dir)
    df = load_processed_dataset(config.input_dir, config.split)
    logging.info("数据加载完成，行数：%d", len(df))

    model = train_quantile_baseline(
        df,
        config.groupby_columns,
        config.value_column,
        config.quantiles,
        config.min_points_per_group,
    )
    logging.info("训练完成，分组数：%d", len(model.table))
    save_quantile_model(model, config.output_dir)
    logging.info("模型已保存：%s", config.output_dir)


if __name__ == "__main__":
    main()
