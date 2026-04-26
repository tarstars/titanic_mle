#!/usr/bin/env python3
"""Study survival of random EML trees under x=0 and x=1 screening."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from titanic_mle.random_eml_trees import (  # noqa: E402
    count_surviving_random_trees,
    first_x_equals_one_overflow_height,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trees", type=int, default=1000)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--scan-max-height", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    screening = count_surviving_random_trees(
        n_trees=args.n_trees,
        height=args.height,
        seed=args.seed,
    )
    scan = [
        asdict(count_surviving_random_trees(n_trees=args.n_trees, height=height, seed=args.seed))
        for height in range(args.scan_max_height + 1)
    ]

    payload = {
        "requested_run": asdict(screening),
        "first_x_equals_one_overflow_height": first_x_equals_one_overflow_height(args.scan_max_height),
        "scan": scan,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
