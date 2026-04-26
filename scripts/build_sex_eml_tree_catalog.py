#!/usr/bin/env python3
"""Build an exhaustive s-expression catalog for EML trees on Titanic sex input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from titanic_mle.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR  # noqa: E402
from titanic_mle.tree_catalog import load_titanic_sex_benchmark_stats, write_catalog  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-height", type=int, default=4)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROCESSED_DATA_DIR / "sex_eml_tree_catalog_height_lt_5.csv.gz",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=PROCESSED_DATA_DIR / "sex_eml_tree_catalog_height_lt_5_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = load_titanic_sex_benchmark_stats(RAW_DATA_DIR / "train.csv")
    summary = write_catalog(
        output_csv_path=args.output_csv,
        output_summary_path=args.output_summary,
        max_height=args.max_height,
        benchmark=benchmark,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
