#!/usr/bin/env python3
"""Analyze calculable unary EML trees for sex and embarked inputs.

This script splits the work into two regimes:

- exact enumeration for all calculable trees with height < 5
- uniform sampling from the exact-height-5 calculable family

The split is necessary because the exact-height-5 family is already very large.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1 / (1 + ez)
    ez = math.exp(z)
    return ez / (1 + ez)


def sexpr(tree: object) -> str:
    if tree in ("x", "1"):
        return str(tree)
    left, right = tree
    return f"({sexpr(left)} {sexpr(right)})"


@lru_cache(None)
def exact_survived(height: int) -> tuple[tuple[object, tuple[float, float, float]], ...]:
    """Return all calculable trees of exact height on x in {0, 0.5, 1}."""

    if height == 0:
        return (("x", (0.0, 0.5, 1.0)), ("1", (1.0, 1.0, 1.0)))

    lower = []
    prev = set(exact_survived(height - 1))
    for h in range(height):
        lower.extend(exact_survived(h))

    out = []
    for left in lower:
        for right in lower:
            if left in prev or right in prev:
                left_tree, left_vals = left
                right_tree, right_vals = right
                try:
                    values = tuple(math.exp(a) - math.log(b) for a, b in zip(left_vals, right_vals))
                    if all(math.isfinite(v) for v in values):
                        out.append(((left_tree, right_tree), values))
                except Exception:
                    pass
    return tuple(out)


def load_benchmarks() -> dict[str, object]:
    """Load grouped Titanic benchmark statistics for sex and embarked."""

    rows = list(csv.DictReader((ROOT / "data" / "raw" / "train.csv").open()))

    male_positive = male_negative = female_positive = female_negative = 0
    embarked_positive = {0.0: 0, 0.5: 0, 1.0: 0}
    embarked_negative = {0.0: 0, 0.5: 0, 1.0: 0}

    for row in rows:
        survived = int(row["Survived"])

        if row["Sex"].strip().lower() == "female":
            if survived:
                female_positive += 1
            else:
                female_negative += 1
        else:
            if survived:
                male_positive += 1
            else:
                male_negative += 1

        embarked = row["Embarked"].strip() or "S"
        embarked_value = {"C": 0.0, "Q": 0.5, "S": 1.0}[embarked]
        if survived:
            embarked_positive[embarked_value] += 1
        else:
            embarked_negative[embarked_value] += 1

    return {
        "total": len(rows),
        "male_positive": male_positive,
        "male_negative": male_negative,
        "female_positive": female_positive,
        "female_negative": female_negative,
        "embarked_positive": embarked_positive,
        "embarked_negative": embarked_negative,
    }


def sex_metrics(benchmark: dict[str, object], score_x0: float, score_x1: float) -> tuple[float, float]:
    """Return ROC AUC and log loss for sex-as-x."""

    male_positive = int(benchmark["male_positive"])
    male_negative = int(benchmark["male_negative"])
    female_positive = int(benchmark["female_positive"])
    female_negative = int(benchmark["female_negative"])
    total = int(benchmark["total"])
    positive_total = male_positive + female_positive
    negative_total = male_negative + female_negative

    wins = 0.0
    if score_x0 > score_x1:
        wins += male_positive * female_negative
    elif score_x0 == score_x1:
        wins += 0.5 * male_positive * female_negative

    if score_x1 > score_x0:
        wins += female_positive * male_negative
    elif score_x1 == score_x0:
        wins += 0.5 * female_positive * male_negative

    wins += 0.5 * male_positive * male_negative
    wins += 0.5 * female_positive * female_negative
    auc = wins / (positive_total * negative_total)

    p0 = min(max(sigmoid(score_x0), 1e-15), 1 - 1e-15)
    p1 = min(max(sigmoid(score_x1), 1e-15), 1 - 1e-15)
    logloss = (
        -male_positive * math.log(p0)
        - male_negative * math.log(1 - p0)
        - female_positive * math.log(p1)
        - female_negative * math.log(1 - p1)
    ) / total
    return auc, logloss


def embarked_metrics(
    benchmark: dict[str, object],
    score_x0: float,
    score_x05: float,
    score_x1: float,
) -> tuple[float, float]:
    """Return ROC AUC and log loss for embarked-as-x."""

    positives = benchmark["embarked_positive"]
    negatives = benchmark["embarked_negative"]
    total = int(benchmark["total"])
    values = [0.0, 0.5, 1.0]
    scores = {0.0: score_x0, 0.5: score_x05, 1.0: score_x1}

    wins = 0.0
    for value_pos in values:
        for value_neg in values:
            count = positives[value_pos] * negatives[value_neg]
            if scores[value_pos] > scores[value_neg]:
                wins += count
            elif scores[value_pos] == scores[value_neg]:
                wins += 0.5 * count

    positive_total = sum(positives.values())
    negative_total = sum(negatives.values())
    auc = wins / (positive_total * negative_total)

    logloss = 0.0
    for value in values:
        probability = min(max(sigmoid(scores[value]), 1e-15), 1 - 1e-15)
        logloss += -positives[value] * math.log(probability)
        logloss += -negatives[value] * math.log(1 - probability)
    return auc, logloss / total


def summarize_exact_less_than_five(benchmark: dict[str, object]) -> dict[str, object]:
    """Exact analysis for all calculable trees with height < 5."""

    rows = []
    for height in range(5):
        for tree, values in exact_survived(height):
            score_x0, score_x05, score_x1 = values
            sex_auc, sex_logloss = sex_metrics(benchmark, score_x0, score_x1)
            embarked_auc, embarked_logloss = embarked_metrics(benchmark, score_x0, score_x05, score_x1)
            rows.append(
                {
                    "sexpr": sexpr(tree),
                    "height": height,
                    "score0": score_x0,
                    "score05": score_x05,
                    "score1": score_x1,
                    "sex_auc": sex_auc,
                    "sex_logloss": sex_logloss,
                    "embarked_auc": embarked_auc,
                    "embarked_logloss": embarked_logloss,
                }
            )

    return {
        "total_survived_lt5": len(rows),
        "reference_sex": {
            "roc_auc": sex_metrics(benchmark, 0.0, 1.0)[0],
            "logloss": sex_metrics(benchmark, 0.0, 1.0)[1],
        },
        "reference_embarked": {
            "roc_auc": embarked_metrics(benchmark, 0.0, 0.5, 1.0)[0],
            "logloss": embarked_metrics(benchmark, 0.0, 0.5, 1.0)[1],
        },
        "top10_sex_auc": sorted(rows, key=lambda row: (-row["sex_auc"], row["sex_logloss"], row["sexpr"]))[:10],
        "top10_sex_logloss": sorted(rows, key=lambda row: (row["sex_logloss"], -row["sex_auc"], row["sexpr"]))[:10],
        "top10_embarked_auc": sorted(
            rows,
            key=lambda row: (-row["embarked_auc"], row["embarked_logloss"], row["sexpr"]),
        )[:10],
        "top10_embarked_logloss": sorted(
            rows,
            key=lambda row: (row["embarked_logloss"], -row["embarked_auc"], row["sexpr"]),
        )[:10],
    }


def sample_exact_height_five(benchmark: dict[str, object], samples: int, seed: int) -> dict[str, object]:
    """Sample from the exact-height-5 calculable family."""

    rng = random.Random(seed)
    exact4 = list(exact_survived(4))
    left_exact4 = [item for item in exact4 if max(item[1]) < 709]
    right_exact4 = [item for item in exact4 if min(item[1]) > 0]

    left_upto3 = []
    right_upto4 = []
    for height in range(4):
        for item in exact_survived(height):
            if max(item[1]) < 709:
                left_upto3.append(item)
    for height in range(5):
        for item in exact_survived(height):
            if min(item[1]) > 0:
                right_upto4.append(item)

    count_a = len(left_exact4) * len(right_upto4)
    count_b = len(left_upto3) * len(right_exact4)
    total_family = count_a + count_b

    sampled = []
    seen = set()
    for _ in range(samples):
        if rng.randrange(total_family) < count_a:
            left_tree, left_values = left_exact4[rng.randrange(len(left_exact4))]
            right_tree, right_values = right_upto4[rng.randrange(len(right_upto4))]
        else:
            left_tree, left_values = left_upto3[rng.randrange(len(left_upto3))]
            right_tree, right_values = right_exact4[rng.randrange(len(right_exact4))]

        expression = f"({sexpr(left_tree)} {sexpr(right_tree)})"
        if expression in seen:
            continue
        seen.add(expression)

        score_x0, score_x05, score_x1 = tuple(
            math.exp(a) - math.log(b) for a, b in zip(left_values, right_values)
        )
        sex_auc, sex_logloss = sex_metrics(benchmark, score_x0, score_x1)
        embarked_auc, embarked_logloss = embarked_metrics(benchmark, score_x0, score_x05, score_x1)

        sampled.append(
            {
                "sexpr": expression,
                "height": 5,
                "score0": score_x0,
                "score05": score_x05,
                "score1": score_x1,
                "sex_auc": sex_auc,
                "sex_logloss": sex_logloss,
                "embarked_auc": embarked_auc,
                "embarked_logloss": embarked_logloss,
            }
        )

    return {
        "exact_height_5_family_size": total_family,
        "sampled_unique_trees": len(sampled),
        "top10_sex_auc": sorted(sampled, key=lambda row: (-row["sex_auc"], row["sex_logloss"], row["sexpr"]))[:10],
        "top10_sex_logloss": sorted(
            sampled,
            key=lambda row: (row["sex_logloss"], -row["sex_auc"], row["sexpr"]),
        )[:10],
        "top10_embarked_auc": sorted(
            sampled,
            key=lambda row: (-row["embarked_auc"], row["embarked_logloss"], row["sexpr"]),
        )[:10],
        "top10_embarked_logloss": sorted(
            sampled,
            key=lambda row: (row["embarked_logloss"], -row["embarked_auc"], row["sexpr"]),
        )[:10],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height5-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260420)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = load_benchmarks()
    payload = {
        "exact_lt5": summarize_exact_less_than_five(benchmark),
        "sampled_h5": sample_exact_height_five(
            benchmark=benchmark,
            samples=args.height5_samples,
            seed=args.seed,
        ),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
