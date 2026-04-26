"""Catalog building for exhaustive EML s-expression trees."""

from __future__ import annotations

import csv
import gzip
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, TextIO

from .metrics import sigmoid
from .sexpr_trees import TreeExpr, build_exact_tree_cache, iter_exact_height_trees, tree_shape_summary, tree_to_sexpr


def evaluate_eml_tree(tree: TreeExpr, x_value: float) -> float:
    """Evaluate an arbitrary EML tree for one x value."""

    if tree == "x":
        return x_value
    if tree == "1":
        return 1.0

    left, right = tree
    left_value = evaluate_eml_tree(left, x_value)
    right_value = evaluate_eml_tree(right, x_value)
    return math.exp(left_value) - math.log(right_value)


@dataclass(frozen=True)
class SexBenchmarkStats:
    """Aggregated Titanic counts for using sex as the only input variable."""

    male_positive: int
    male_negative: int
    female_positive: int
    female_negative: int

    @property
    def total(self) -> int:
        return self.male_positive + self.male_negative + self.female_positive + self.female_negative

    @property
    def positive_total(self) -> int:
        return self.male_positive + self.female_positive

    @property
    def negative_total(self) -> int:
        return self.male_negative + self.female_negative

    def roc_auc_from_scores(self, score_x0: float, score_x1: float) -> float:
        """Compute ROC AUC induced by scores for males (x=0) and females (x=1)."""

        if score_x1 > score_x0:
            wins = (
                self.female_positive * self.male_negative
                + 0.5 * self.male_positive * self.male_negative
                + 0.5 * self.female_positive * self.female_negative
            )
        elif score_x1 < score_x0:
            wins = (
                self.male_positive * self.female_negative
                + 0.5 * self.male_positive * self.male_negative
                + 0.5 * self.female_positive * self.female_negative
            )
        else:
            wins = 0.5 * self.positive_total * self.negative_total

        return wins / (self.positive_total * self.negative_total)

    def logloss_from_scores(self, score_x0: float, score_x1: float, eps: float = 1e-15) -> float:
        """Compute log loss induced by scores for males (x=0) and females (x=1)."""

        prob_x0 = min(max(sigmoid(score_x0), eps), 1.0 - eps)
        prob_x1 = min(max(sigmoid(score_x1), eps), 1.0 - eps)

        total = (
            -self.male_positive * math.log(prob_x0)
            - self.male_negative * math.log(1.0 - prob_x0)
            - self.female_positive * math.log(prob_x1)
            - self.female_negative * math.log(1.0 - prob_x1)
        )
        return total / self.total


def load_titanic_sex_benchmark_stats(train_csv_path: Path) -> SexBenchmarkStats:
    """Load aggregated Titanic counts for the sex-as-x experiment."""

    male_positive = male_negative = female_positive = female_negative = 0
    with train_csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            is_female = row["Sex"].strip().lower() == "female"
            survived = int(row["Survived"])
            if is_female and survived == 1:
                female_positive += 1
            elif is_female and survived == 0:
                female_negative += 1
            elif not is_female and survived == 1:
                male_positive += 1
            else:
                male_negative += 1

    return SexBenchmarkStats(
        male_positive=male_positive,
        male_negative=male_negative,
        female_positive=female_positive,
        female_negative=female_negative,
    )


def build_catalog_row(tree: TreeExpr, benchmark: SexBenchmarkStats) -> dict[str, object]:
    """Build one catalog row for a tree."""

    summary = tree_shape_summary(tree)
    row: dict[str, object] = {
        "sexpr": tree_to_sexpr(tree),
        "height": summary.height,
        "leaf_count": summary.leaf_count,
        "internal_node_count": summary.internal_node_count,
        "total_node_count": summary.total_node_count,
        "survives_x0": False,
        "survives_x1": False,
        "survives_binary_input": False,
        "error_x0": "",
        "error_x1": "",
        "score_x0": "",
        "score_x1": "",
        "roc_auc_sex_input": "",
        "logloss_sex_input": "",
    }

    score_x0: float | None = None
    score_x1: float | None = None

    try:
        score_x0 = evaluate_eml_tree(tree, 0.0)
        row["survives_x0"] = True
        row["score_x0"] = score_x0
    except Exception as exc:
        row["error_x0"] = type(exc).__name__

    try:
        score_x1 = evaluate_eml_tree(tree, 1.0)
        row["survives_x1"] = True
        row["score_x1"] = score_x1
    except Exception as exc:
        row["error_x1"] = type(exc).__name__

    if score_x0 is not None and score_x1 is not None:
        row["survives_binary_input"] = True
        row["roc_auc_sex_input"] = benchmark.roc_auc_from_scores(score_x0, score_x1)
        row["logloss_sex_input"] = benchmark.logloss_from_scores(score_x0, score_x1)

    return row


def iter_catalog_rows(max_height: int, benchmark: SexBenchmarkStats) -> Iterator[dict[str, object]]:
    """Yield catalog rows for all trees with height <= max_height."""

    if max_height < 0:
        raise ValueError("max_height must be non-negative")

    if max_height == 0:
        for tree in ("x", "1"):
            yield build_catalog_row(tree, benchmark)
        return

    cache = build_exact_tree_cache(max_height - 1)
    for height in range(max_height):
        for tree in cache[height]:
            yield build_catalog_row(tree, benchmark)

    for tree in iter_exact_height_trees(max_height, cache):
        yield build_catalog_row(tree, benchmark)


def summarize_catalog_rows(rows: Iterator[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Consume rows and return them plus a summary object."""

    materialized = []
    counts_by_height: dict[int, dict[str, int]] = {}

    for row in rows:
        materialized.append(row)
        height = int(row["height"])
        bucket = counts_by_height.setdefault(height, {"total": 0, "survived": 0})
        bucket["total"] += 1
        if row["survives_binary_input"]:
            bucket["survived"] += 1

    summary = {
        "counts_by_exact_height": {
            str(height): counts for height, counts in sorted(counts_by_height.items())
        },
        "total_trees": len(materialized),
        "survived_trees": sum(1 for row in materialized if row["survives_binary_input"]),
    }
    return materialized, summary


def _open_text_writer(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def write_catalog(
    output_csv_path: Path,
    output_summary_path: Path,
    max_height: int,
    benchmark: SexBenchmarkStats,
) -> dict[str, object]:
    """Write a full catalog and summary for trees up to max_height."""

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    counts_by_height: dict[int, dict[str, int]] = {}
    total_trees = 0
    survived_trees = 0

    with _open_text_writer(output_csv_path) as handle:
        writer: csv.DictWriter | None = None

        for row in iter_catalog_rows(max_height=max_height, benchmark=benchmark):
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                writer.writeheader()

            writer.writerow(row)
            total_trees += 1
            height = int(row["height"])
            bucket = counts_by_height.setdefault(height, {"total": 0, "survived": 0})
            bucket["total"] += 1
            if row["survives_binary_input"]:
                bucket["survived"] += 1
                survived_trees += 1

    summary_payload = {
        "max_height": max_height,
        "benchmark": asdict(benchmark),
        "counts_by_exact_height": {
            str(height): counts for height, counts in sorted(counts_by_height.items())
        },
        "total_trees": total_trees,
        "survived_trees": survived_trees,
    }
    output_summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    return summary_payload
