#!/usr/bin/env python3
from pathlib import Path
import sys
import argparse
import json
import csv
import logging

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.tcn_infer import infer_single, load_infer_config_from_dict  # noqa: E402

SITE_CURRENCY_MAP = {
    "US": "USD",
    "UK": "GBP",
    "DE": "EUR",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/tcn/infer.json"))
    parser.add_argument("--site", type=str, default=None)
    parser.add_argument("--currency", type=str, default=None)
    parser.add_argument("--fee-type", type=str, default=None)
    parser.add_argument("--series-id", type=str, default=None)
    parser.add_argument("--target-step", type=int, default=None)
    args = parser.parse_args()
    # 推理入口，默认读取 TCN 推理配置。
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    filters = payload.get("filters") or {}
    if args.site is not None:
        filters["site"] = args.site
        if args.currency is None:
            mapped = SITE_CURRENCY_MAP.get(args.site.upper())
            if mapped:
                filters["currency"] = mapped
    if args.currency is not None:
        filters["currency"] = args.currency
    if args.fee_type is not None:
        filters["fee_type"] = args.fee_type
    if args.series_id is not None:
        filters["series_id"] = args.series_id
    payload["filters"] = filters
    if args.target_step is not None:
        payload["target_step"] = args.target_step

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_infer_config_from_dict(payload)
    record = infer_single(config)
    output_path = config.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
        writer.writeheader()
        writer.writerow(record)
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("推理完成，输出：%s", output_path)
    logging.info("推理结果：%s", json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
