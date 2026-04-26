#!/usr/bin/env python3
"""Analyze survived height-2 EML trees on Titanic using sex as x."""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from titanic_mle.metrics import binary_logloss, roc_auc_score, sigmoid  # noqa: E402
from titanic_mle.random_eml_trees import evaluate_full_binary_eml_tree  # noqa: E402


def load_titanic_sex_target() -> tuple[list[int], list[float]]:
    """Load survival labels and sex-as-x values from Titanic train.csv."""

    path = ROOT / "data" / "raw" / "train.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    y_true = [int(row["Survived"]) for row in rows]
    x_values = [1.0 if row["Sex"].strip().lower() == "female" else 0.0 for row in rows]
    return y_true, x_values


def survived_height2_leaf_assignments() -> list[tuple[str, str, str, str]]:
    """Enumerate unique survived full-binary trees of height 2."""

    survived = []
    for leaves in itertools.product(("x", "1"), repeat=4):
        try:
            evaluate_full_binary_eml_tree(list(leaves), 0.0)
            evaluate_full_binary_eml_tree(list(leaves), 1.0)
        except Exception:
            continue
        survived.append(leaves)
    return survived


def analyze_height2_trees() -> dict[str, object]:
    """Return a ranking report for survived height-2 trees."""

    y_true, x_values = load_titanic_sex_target()
    survived = survived_height2_leaf_assignments()

    results = []
    for leaves in survived:
        scores = [evaluate_full_binary_eml_tree(list(leaves), x_value) for x_value in x_values]
        probabilities = [sigmoid(score) for score in scores]
        result = {
            "leaves": "".join(leaves),
            "score_x0": evaluate_full_binary_eml_tree(list(leaves), 0.0),
            "score_x1": evaluate_full_binary_eml_tree(list(leaves), 1.0),
            "roc_auc": roc_auc_score(y_true, scores),
            "logloss": binary_logloss(y_true, probabilities),
        }
        results.append(result)

    top_by_auc = sorted(results, key=lambda item: (-item["roc_auc"], item["logloss"], item["leaves"]))
    top_by_logloss = sorted(results, key=lambda item: (item["logloss"], -item["roc_auc"], item["leaves"]))

    return {
        "survived_unique_trees": len(results),
        "reference_sex_roc_auc": roc_auc_score(y_true, x_values),
        "top_by_roc_auc": top_by_auc[:10],
        "top_by_logloss": top_by_logloss[:10],
    }


def main() -> None:
    print(json.dumps(analyze_height2_trees(), indent=2))


if __name__ == "__main__":
    main()
