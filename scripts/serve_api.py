#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402


def main() -> None:
    uvicorn.run("src.service.app:app", host="0.0.0.0", port=8088, reload=False)


if __name__ == "__main__":
    main()
