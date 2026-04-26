#!/usr/bin/env python3
"""Count valid two-input EML trees for the sex/embarked experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from titanic_mle.two_input_discrete_trees import (  # noqa: E402
    count_valid_two_input_trees_height_le_five,
)


def main() -> None:
    summary = count_valid_two_input_trees_height_le_five()
    print(
        json.dumps(
            {
                "domain": summary.domain,
                "exact_height_counts": summary.exact_height_counts,
                "unique_signature_counts": summary.unique_signature_counts,
                "left_safe_counts": summary.left_safe_counts,
                "right_safe_counts": summary.right_safe_counts,
                "total_count_le_5": summary.total_count_le_5,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
