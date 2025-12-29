#!/usr/bin/env python3
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.baseline import save_quantile_model, train_quantile_baseline  # noqa: E402
from src.training.config import load_config  # noqa: E402
from src.training.data import load_processed_dataset  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 读取训练配置与数据。
    config = load_config(Path("configs/baseline/train.json"))
    logging.info("读取配置：%s", Path("configs/baseline/train.json"))
    logging.info("加载数据：%s", config.input_dir)
    df = load_processed_dataset(config.input_dir, config.split)
    logging.info("数据加载完成，行数：%d", len(df))

    # 训练分位数基线模型。
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
