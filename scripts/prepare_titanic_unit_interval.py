#!/usr/bin/env python3
"""Generate unit-interval Titanic feature tables for EML experiments."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from titanic_mle.preprocessing import prepare_titanic_unit_interval


def main() -> None:
    outputs = prepare_titanic_unit_interval()
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
