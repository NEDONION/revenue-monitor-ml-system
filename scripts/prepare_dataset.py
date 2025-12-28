#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import logging


REQUIRED_COLUMNS = {
    "ts",
    "granularity",
    "site",
    "currency",
    "fee_type",
    "metric_value",
}

# 默认窗口与滞后配置，按粒度提供经验值。
DEFAULT_WINDOWS_BY_GRANULARITY = {
    "minute": [60, 240, 1440],
    "hourly": [24, 168, 336],
    "daily": [7, 14, 28],
}

DEFAULT_LAGS_BY_GRANULARITY = {
    "minute": [1, 60, 1440],
    "hourly": [1, 24, 168],
    "daily": [1, 7, 14, 28],
}

# 促销相关的默认参数（固定在代码内）。
PROMO_RATE = 0.01
PROMO_TYPES = ["BlackFriday", "FlashSale", "FeeWaiver"]
DEFAULT_RANDOM_SEED = 42

# 站点到时区的映射，用于生成本地时间特征。
SITE_TIMEZONE_MAP = {
    "US": "America/Los_Angeles",
    "UK": "Europe/London",
    "DE": "Europe/Berlin",
}

# 简化节假日定义（可按需扩展），仅按本地日期月/日匹配。
# 注意：不含浮动节假日（如感恩节、复活节等），这里只做流程演示。
HOLIDAYS_BY_SITE = {
    "US": [
        (1, 1),   # New Year's Day
        (1, 15),  # Martin Luther King Jr. Day (fixed fallback)
        (2, 19),  # Presidents' Day (fixed fallback)
        (5, 27),  # Memorial Day (fixed fallback)
        (6, 19),  # Juneteenth
        (7, 4),   # Independence Day
        (9, 2),   # Labor Day (fixed fallback)
        (10, 14), # Columbus Day (fixed fallback)
        (11, 11), # Veterans Day
        (11, 28), # Thanksgiving (fixed fallback)
        (12, 25), # Christmas
    ],
    "UK": [
        (1, 1),   # New Year's Day
        (4, 1),   # Easter Monday (fixed fallback)
        (5, 6),   # Early May Bank Holiday (fixed fallback)
        (5, 27),  # Spring Bank Holiday (fixed fallback)
        (8, 26),  # Summer Bank Holiday (fixed fallback)
        (12, 25), # Christmas
        (12, 26), # Boxing Day
    ],
    "DE": [
        (1, 1),   # New Year's Day
        (5, 1),   # Labour Day
        (10, 3),  # German Unity Day
        (12, 25), # Christmas Day
        (12, 26), # Second Day of Christmas
        (1, 6),   # Epiphany (some states)
        (8, 15),  # Assumption Day (some states)
        (10, 31), # Reformation Day (some states)
        (11, 1),  # All Saints' Day (some states)
    ],
}


@dataclass(frozen=True)
class SplitConfig:
    train: float
    val: float
    test: float


def parse_csv_list(value: Optional[str]) -> Optional[List[int]]:
    # 解析逗号分隔的整数列表参数（如 "7,14,28"）。
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_split(value: str) -> SplitConfig:
    # 解析训练/验证/测试切分比例。
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("split must be in the form 'train,val,test'")
    total = sum(parts)
    if abs(total - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")
    return SplitConfig(*parts)


def read_input(path: Path) -> pd.DataFrame:
    # 读取输入文件（CSV/Parquet），并在缺失时给出可用候选。
    if not path.exists():
        # 当输入路径不存在时，尝试在同目录下匹配常见扩展名。
        candidates = [
            path.with_suffix(".csv"),
            path.with_suffix(".parquet"),
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
        else:
            existing = sorted(p.name for p in path.parent.glob("*"))
            raise FileNotFoundError(
                f"Input file not found: {path}. "
                f"Available files in {path.parent}: {existing}"
            )
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def create_sample_dataset(path: Path, granularity: str, sample_rows: int) -> Path:
    # 生成一个可跑通流程的最小样例数据集。
    if granularity not in ("minute", "hourly", "daily"):
        granularity = "daily"
    if path.exists() and path.is_dir():
        # 若路径被目录占用，直接删除并用文件替换。
        shutil.rmtree(path)
    freq = freq_from_granularity(granularity)
    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    total_rows = max(1, sample_rows)
    sites = ["US", "UK", "DE"]
    currencies = ["USD", "GBP", "EUR"]
    fee_types = ["listing_fee", "final_value_fee", "payment_processing_fee"]
    currency_fx = {"USD": 1.0, "GBP": 1.25, "EUR": 1.08}
    fee_base = {
        "listing_fee": 0.3,
        "final_value_fee": 2.5,
        "payment_processing_fee": 0.8,
    }
    base_lambda = {
        "listing_fee": 12,
        "final_value_fee": 25,
        "payment_processing_fee": 18,
    }

    series = []
    series_keys = []
    for site in sites:
        for currency in currencies:
            for fee_type in fee_types:
                series_keys.append((site, currency, fee_type))

    points_per_series = max(1, total_rows // len(series_keys))
    ts = pd.date_range("2024-01-01", periods=points_per_series, freq=freq, tz="UTC")

    for site, currency, fee_type in series_keys:
        seasonal = 1.0 + 0.2 * np.sin(np.linspace(0, 6 * np.pi, len(ts)))
        noise = rng.normal(0, 0.05, size=len(ts))
        avg_fee = fee_base[fee_type] * (1 + rng.normal(0, 0.1))
        lam = base_lambda[fee_type] * seasonal
        tx_count = rng.poisson(lam=lam).astype(int)
        metric_value = np.maximum(
            0.0,
            tx_count * avg_fee * (1 + noise) + rng.normal(0, 0.5, size=len(ts)),
        ).astype(float)
        series.append(
            pd.DataFrame(
                {
                    "ts": ts,
                    "granularity": granularity,
                    "site": site,
                    "currency": currency,
                    "fee_type": fee_type,
                    "metric_name": "revenue",
                    "metric_value": metric_value,
                    "tx_count": tx_count,
                    "fx_rate_to_usd": currency_fx[currency],
                    "timezone": SITE_TIMEZONE_MAP.get(site, "UTC"),
                }
            )
        )

    df = pd.concat(series, ignore_index=True).head(total_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    # 校验必需字段，并补齐 metric_name / series_id。
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "metric_name" not in df.columns:
        df = df.copy()
        df["metric_name"] = "revenue"
    if "series_id" not in df.columns:
        df = df.copy()
        # 构建稳定的序列标识，确保每条时间序列可独立处理。
        df["series_id"] = (
            df["site"].astype(str)
            + "|"
            + df["currency"].astype(str)
            + "|"
            + df["fee_type"].astype(str)
            + "|"
            + df["metric_name"].astype(str)
            + "|"
            + df["granularity"].astype(str)
        )
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    # 去重策略：同一序列同一时间点保留最新记录。
    if "ingest_ts" in df.columns:
        df = df.sort_values(["series_id", "ts", "ingest_ts"])
    else:
        df = df.sort_values(["series_id", "ts"])
    # 对每个 (series_id, ts) 保留最新记录，避免重复点影响训练。
    return df.drop_duplicates(subset=["series_id", "ts"], keep="last")


def freq_from_granularity(granularity: str) -> str:
    # 将粒度映射为 pandas 频率字符串。
    mapping = {"minute": "min", "hourly": "h", "daily": "D"}
    if granularity not in mapping:
        raise ValueError(f"Unsupported granularity: {granularity}")
    return mapping[granularity]


def fill_time_axis(
    df: pd.DataFrame, granularity: str, missing_strategy: str
) -> pd.DataFrame:
    # 按粒度补齐时间轴，并根据策略处理缺失值。
    freq = freq_from_granularity(granularity)
    df = df.sort_values("ts")
    # 记录维度字段的默认值，避免补齐时间轴后出现空值。
    dim_cols = ["series_id", "granularity", "site", "currency", "fee_type", "metric_name"]
    dim_defaults: Dict[str, str] = {}
    for col in dim_cols:
        if col in df.columns and df[col].notna().any():
            dim_defaults[col] = df[col].dropna().iloc[0]
        else:
            dim_defaults[col] = "unknown"

    # 重建完整时间轴，标记缺失点。
    index = pd.date_range(df["ts"].min(), df["ts"].max(), freq=freq)
    df = df.set_index("ts").reindex(index)
    df.index.name = "ts"

    for col in dim_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
            df[col] = df[col].fillna(dim_defaults[col])

    df["is_missing"] = df["metric_value"].isna()

    if missing_strategy == "zero":
        df["metric_value"] = df["metric_value"].fillna(0)
    elif missing_strategy == "ffill":
        df["metric_value"] = df["metric_value"].ffill().fillna(0)
    elif missing_strategy == "interpolate":
        df["metric_value"] = (
            df["metric_value"]
            .interpolate(method="time", limit_direction="forward")
            .ffill()
            .fillna(0)
        )
    else:
        raise ValueError(f"Unknown missing strategy: {missing_strategy}")

    return df.reset_index()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # 时间特征有助于模型学习季节性与周期性。
    ts = df["ts"]
    df["local_ts"] = ts
    if "timezone" in df.columns:
        def _to_local(row_ts, tz):
            if pd.isna(tz):
                return row_ts
            try:
                return row_ts.tz_convert(tz)
            except Exception:
                return row_ts
        df["local_ts"] = [
            _to_local(row_ts, tz) for row_ts, tz in zip(ts, df["timezone"])
        ]
    local_ts = df["local_ts"]
    df["hour"] = ts.dt.hour
    df["local_hour"] = local_ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_month"] = ts.dt.day
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_context_features(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    # 补充业务上下文特征：时区、节假日、促销标记。
    # 补充时区字段（若缺失则按 site 映射）。
    if "timezone" not in df.columns:
        df["timezone"] = df["site"].map(SITE_TIMEZONE_MAP).fillna("UTC")

    # 生成节假日标记（基于本地日期的简化规则）。
    if "is_holiday" not in df.columns:
        local_date = df["local_ts"].dt.date
        month = df["local_ts"].dt.month
        day = df["local_ts"].dt.day
        is_holiday = []
        for site, m, d in zip(df["site"], month, day):
            holidays = HOLIDAYS_BY_SITE.get(site, [])
            is_holiday.append((m, d) in holidays)
        df["is_holiday"] = pd.Series(is_holiday, index=df.index).astype(int)

    # 生成促销标记与类型（随机，便于流程联调）。
    if "is_promo" not in df.columns:
        df["is_promo"] = rng.random(len(df)) < PROMO_RATE
    if "promo_type" not in df.columns:
        if PROMO_TYPES:
            choices = rng.choice(PROMO_TYPES, size=len(df))
            df["promo_type"] = np.where(df["is_promo"], choices, "")
        else:
            df["promo_type"] = ""

    return df


def add_window_features(
    df: pd.DataFrame, windows: Iterable[int], lags: Iterable[int]
) -> Tuple[pd.DataFrame, List[str]]:
    # 滚动统计与滞后特征用于刻画局部趋势与周期。
    df = df.sort_values("ts")
    feature_cols: List[str] = []
    shifted = df["metric_value"].shift(1)
    for window in windows:
        mean_col = f"rolling_mean_{window}"
        std_col = f"rolling_std_{window}"
        min_col = f"rolling_min_{window}"
        max_col = f"rolling_max_{window}"
        df[mean_col] = shifted.rolling(window, min_periods=1).mean()
        df[std_col] = shifted.rolling(window, min_periods=1).std()
        df[min_col] = shifted.rolling(window, min_periods=1).min()
        df[max_col] = shifted.rolling(window, min_periods=1).max()
        feature_cols.extend([mean_col, std_col, min_col, max_col])
    for lag in lags:
        lag_col = f"lag_{lag}"
        df[lag_col] = df["metric_value"].shift(lag)
        feature_cols.append(lag_col)
    return df, feature_cols


def assign_split(df: pd.DataFrame, split: SplitConfig) -> pd.DataFrame:
    # 以时间顺序做训练/验证/测试切分。
    unique_ts = df["ts"].sort_values().unique()
    if len(unique_ts) < 3:
        df["split"] = "train"
        return df
    # 按时间切分，避免信息泄露。
    train_cut = unique_ts[int(len(unique_ts) * split.train) - 1]
    val_cut = unique_ts[int(len(unique_ts) * (split.train + split.val)) - 1]
    df["split"] = "test"
    df.loc[df["ts"] <= val_cut, "split"] = "val"
    df.loc[df["ts"] <= train_cut, "split"] = "train"
    return df


def process_granularity(
    df: pd.DataFrame,
    granularity: str,
    missing_strategy: str,
    windows: List[int],
    lags: List[int],
    split: SplitConfig,
    drop_feature_na: bool,
    rng: np.random.Generator,
) -> pd.DataFrame:
    # 对单一粒度的数据进行补齐、特征构造与切分。
    parts = []
    for _, group in df.groupby("series_id"):
        aligned = fill_time_axis(group, granularity, missing_strategy)
        aligned = add_time_features(aligned)
        aligned = add_context_features(aligned, rng)
        if "tx_count" in aligned.columns:
            aligned["avg_value"] = aligned["metric_value"] / aligned["tx_count"].clip(lower=1)
        aligned, feature_cols = add_window_features(aligned, windows, lags)
        if drop_feature_na:
            aligned = aligned.dropna(subset=feature_cols)
        parts.append(aligned)
    result = pd.concat(parts, ignore_index=True)
    result = assign_split(result, split)
    return result


def write_dataset(df: pd.DataFrame, output_dir: Path, name: str) -> None:
    # 输出到 Parquet（失败则回退到 CSV）。
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{name}.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        csv_path = output_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # CLI 入口：读取输入、生成特征、输出训练数据。
    parser = argparse.ArgumentParser(description="Prepare training dataset from revenue_ts_wide.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/revenue_ts_wide.csv"),
        help="Input CSV/Parquet file.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--granularity", choices=["minute", "hourly", "daily"], default="hourly")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--missing-strategy",
        choices=["zero", "ffill", "interpolate"],
        default="ffill",
    )
    parser.add_argument("--windows", type=str, default=None, help="Comma-separated window sizes.")
    parser.add_argument("--lags", type=str, default=None, help="Comma-separated lag sizes.")
    parser.add_argument(
        "--split",
        type=parse_split,
        default=parse_split("0.7,0.15,0.15"),
        help="Train/val/test ratios, e.g. 0.7,0.15,0.15",
    )
    parser.add_argument(
        "--write-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write train/val/test files separately.",
    )
    parser.add_argument(
        "--drop-feature-na",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop rows with NaN lag/rolling features.",
    )
    parser.add_argument("--sample-rows", type=int, default=1000000)
    args = parser.parse_args()

    if not args.input.exists():
        sample_path = create_sample_dataset(args.input, args.granularity or "daily", args.sample_rows)
        logging.info("未找到输入文件，已生成样例数据：%s", sample_path)

    logging.info("开始读取输入：%s", args.input)
    df = read_input(args.input)
    df = ensure_schema(df)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    logging.info("字段校验完成，当前行数：%d", len(df))

    if args.granularity:
        df = df[df["granularity"] == args.granularity]
        logging.info("按粒度过滤：%s，剩余行数：%d", args.granularity, len(df))

    if args.require_complete and "is_complete" in df.columns:
        df = df[df["is_complete"] == True]  # noqa: E712
        logging.info("应用 is_complete 过滤，剩余行数：%d", len(df))

    df = deduplicate(df)
    logging.info("去重完成，当前行数：%d", len(df))

    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)

    granularities = [args.granularity] if args.granularity else df["granularity"].unique()
    for granularity in granularities:
        subset = df[df["granularity"] == granularity].copy()
        if subset.empty:
            logging.warning("该粒度无数据，已跳过：%s", granularity)
            continue
        windows = parse_csv_list(args.windows) or DEFAULT_WINDOWS_BY_GRANULARITY.get(
            granularity, [7, 14, 28]
        )
        lags = parse_csv_list(args.lags) or DEFAULT_LAGS_BY_GRANULARITY.get(
            granularity, [1, 7, 14, 28]
        )
        prepared = process_granularity(
            subset,
            granularity,
            args.missing_strategy,
            windows,
            lags,
            args.split,
            args.drop_feature_na,
            rng,
        )
        name = f"dataset_{granularity}"
        if args.write_splits:
            for split_name, split_df in prepared.groupby("split"):
                logging.info(
                    "写出数据：%s 分片，行数：%d，输出目录：%s",
                    split_name,
                    len(split_df),
                    args.output_dir,
                )
                write_dataset(split_df, args.output_dir, f"{name}_{split_name}")
        else:
            logging.info("写出数据：行数：%d，输出目录：%s", len(prepared), args.output_dir)
            write_dataset(prepared, args.output_dir, name)

    logging.info("数据集构造完成。")


if __name__ == "__main__":
    main()
