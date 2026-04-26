#!/usr/bin/env python3
"""Benchmark CatBoost ROC AUC on the normalized Titanic train set."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catboost import CatBoostClassifier

from titanic_mle.metrics import binary_logloss, roc_auc_score


TRAIN_PATH = ROOT / "data" / "interim" / "titanic_unit_interval_train.csv"
STACKED_REPORT_PATH = ROOT / "data" / "processed" / "meta_stacked_exact_search_top5_height_le_3.json"
OUTPUT_PATH = ROOT / "data" / "processed" / "catboost_unit_interval_benchmark.json"


@dataclass(frozen=True)
class BenchmarkResult:
    feature_set_name: str
    feature_names: list[str]
    train_auc: float
    train_logloss: float
    oof_auc: float
    oof_logloss: float
    mean_fold_auc: float
    fold_aucs: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20_260_420)
    return parser.parse_args()


def load_train_rows() -> tuple[list[dict[str, float]], list[int], list[str]]:
    with TRAIN_PATH.open() as handle:
        reader = csv.DictReader(handle)
        rows = []
        labels = []
        for row in reader:
            parsed = {key: float(value) for key, value in row.items() if key != "Survived"}
            rows.append(parsed)
            labels.append(int(row["Survived"]))

    feature_names = [name for name in rows[0].keys() if name != "PassengerId"]
    return rows, labels, feature_names


def build_matrix(rows: list[dict[str, float]], feature_names: list[str]) -> list[list[float]]:
    return [[row[name] for name in feature_names] for row in rows]


def fit_catboost(
    x_train: list[list[float]],
    y_train: list[int],
    x_eval: list[list[float]],
    args: argparse.Namespace,
) -> list[float]:
    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=args.seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_eval)[:, 1]
    return [float(probability) for probability in probabilities]


def stratified_folds(labels: list[int], n_folds: int, seed: int) -> list[list[int]]:
    positives = [index for index, label in enumerate(labels) if label == 1]
    negatives = [index for index, label in enumerate(labels) if label == 0]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    folds = [[] for _ in range(n_folds)]
    for bucket, indices in ((folds, positives), (folds, negatives)):
        for offset, index in enumerate(indices):
            bucket[offset % n_folds].append(index)

    for fold in folds:
        fold.sort()
    return folds


def benchmark_feature_set(
    name: str,
    feature_names: list[str],
    rows: list[dict[str, float]],
    labels: list[int],
    args: argparse.Namespace,
) -> BenchmarkResult:
    matrix = build_matrix(rows, feature_names)
    train_probabilities = fit_catboost(matrix, labels, matrix, args)
    train_auc = roc_auc_score(labels, train_probabilities)
    train_logloss = binary_logloss(labels, train_probabilities)

    folds = stratified_folds(labels, args.folds, args.seed)
    oof_probabilities = [0.0] * len(labels)
    fold_aucs: list[float] = []

    for fold_indices in folds:
        validation_set = set(fold_indices)
        train_x = [row for index, row in enumerate(matrix) if index not in validation_set]
        train_y = [label for index, label in enumerate(labels) if index not in validation_set]
        valid_x = [matrix[index] for index in fold_indices]
        valid_y = [labels[index] for index in fold_indices]

        fold_probabilities = fit_catboost(train_x, train_y, valid_x, args)
        for index, probability in zip(fold_indices, fold_probabilities):
            oof_probabilities[index] = probability
        fold_aucs.append(roc_auc_score(valid_y, fold_probabilities))

    oof_auc = roc_auc_score(labels, oof_probabilities)
    oof_logloss = binary_logloss(labels, oof_probabilities)

    return BenchmarkResult(
        feature_set_name=name,
        feature_names=feature_names,
        train_auc=train_auc,
        train_logloss=train_logloss,
        oof_auc=oof_auc,
        oof_logloss=oof_logloss,
        mean_fold_auc=sum(fold_aucs) / len(fold_aucs),
        fold_aucs=fold_aucs,
    )


def load_current_eml_frontier() -> dict[str, object]:
    with STACKED_REPORT_PATH.open() as handle:
        report = json.load(handle)
    return {
        "source_report": str(STACKED_REPORT_PATH.relative_to(ROOT)),
        "best_stacked_auc_expr": report["best_by_auc"]["expr"],
        "best_stacked_auc": report["best_by_auc"]["auc"],
        "best_stacked_auc_logloss": report["best_by_auc"]["logloss"],
        "best_stacked_logloss_expr": report["best_by_logloss"]["expr"],
        "best_stacked_logloss_auc": report["best_by_logloss"]["auc"],
        "best_stacked_calibrated_logloss": report["best_by_logloss"]["calibrated_logloss"],
    }


def main() -> None:
    args = parse_args()
    rows, labels, all_unit_features = load_train_rows()
    results = [
        benchmark_feature_set(
            name="all_unit_features",
            feature_names=all_unit_features,
            rows=rows,
            labels=labels,
            args=args,
        ),
        benchmark_feature_set(
            name="stacked_support_features",
            feature_names=["pclass_unit", "sex_unit", "age_unit", "fare_unit"],
            rows=rows,
            labels=labels,
            args=args,
        ),
    ]

    eml_frontier = load_current_eml_frontier()
    payload = {
        "train_dataset": str(TRAIN_PATH.relative_to(ROOT)),
        "rows": len(rows),
        "seed": args.seed,
        "catboost_params": {
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "folds": args.folds,
        },
        "eml_frontier": eml_frontier,
        "benchmarks": [
            {
                "feature_set_name": result.feature_set_name,
                "feature_names": result.feature_names,
                "train_auc": result.train_auc,
                "train_logloss": result.train_logloss,
                "oof_auc": result.oof_auc,
                "oof_logloss": result.oof_logloss,
                "mean_fold_auc": result.mean_fold_auc,
                "fold_aucs": result.fold_aucs,
                "delta_vs_best_stacked_auc_train": result.train_auc - eml_frontier["best_stacked_auc"],
                "delta_vs_best_stacked_auc_oof": result.oof_auc - eml_frontier["best_stacked_auc"],
            }
            for result in results
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))

    print(f"saved {OUTPUT_PATH}")
    print(f"best stacked EML ROC AUC: {eml_frontier['best_stacked_auc']:.16f}")
    for result in results:
        print(
            f"{result.feature_set_name}: "
            f"train_auc={result.train_auc:.16f} "
            f"oof_auc={result.oof_auc:.16f} "
            f"train_logloss={result.train_logloss:.16f} "
            f"oof_logloss={result.oof_logloss:.16f}"
        )


if __name__ == "__main__":
    main()
