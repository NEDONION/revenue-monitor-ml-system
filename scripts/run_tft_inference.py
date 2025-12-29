#!/usr/bin/env python3
from pathlib import Path
import sys
import argparse
import json
import logging

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.tft.infer import infer_tft_from_dict  # noqa: E402

SITE_CURRENCY_MAP = {
    "US": "USD",
    "UK": "GBP",
    "DE": "EUR",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/tft/infer.json"))
    parser.add_argument("--site", type=str, default=None)
    parser.add_argument("--currency", type=str, default=None)
    parser.add_argument("--fee-type", type=str, default=None)
    parser.add_argument("--series-id", type=str, default=None)
    parser.add_argument("--target-step", type=int, default=None)
    args = parser.parse_args()

    config_path = args.config
    payload = json.loads(config_path.read_text(encoding="utf-8"))
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
    result = infer_tft_from_dict(payload)
    logging.info("推理结果：%s", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
