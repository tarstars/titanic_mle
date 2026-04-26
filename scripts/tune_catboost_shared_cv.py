#!/usr/bin/env python3
"""Tune CatBoost on the normalized Titanic train set with shared CV."""

from __future__ import annotations

import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from titanic_mle.metrics import binary_logloss, roc_auc_score


TRAIN_PATH = ROOT / "data" / "interim" / "titanic_unit_interval_train.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "catboost_shared_cv_tuning.json"

SEED = 20_260_420
FOLDS = 5
THREAD_COUNT = 8


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    params: dict[str, object]
    train_auc: float
    train_logloss: float
    oof_auc: float
    oof_logloss: float
    mean_fold_auc: float
    fold_aucs: list[float]
    auc_gap: float
    logloss_gap: float
    balanced_score: float


def load_train_rows() -> tuple[list[list[float]], list[int], list[str]]:
    with TRAIN_PATH.open() as handle:
        reader = csv.DictReader(handle)
        matrix = []
        labels = []
        feature_names: list[str] | None = None
        for row in reader:
            if feature_names is None:
                feature_names = [name for name in row.keys() if name not in {"PassengerId", "Survived"}]
            matrix.append([float(row[name]) for name in feature_names])
            labels.append(int(row["Survived"]))
    assert feature_names is not None
    return matrix, labels, feature_names


def stratified_folds(labels: list[int], n_folds: int, seed: int) -> list[list[int]]:
    positives = [index for index, label in enumerate(labels) if label == 1]
    negatives = [index for index, label in enumerate(labels) if label == 0]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    folds = [[] for _ in range(n_folds)]
    for indices in (positives, negatives):
        for offset, index in enumerate(indices):
            folds[offset % n_folds].append(index)

    for fold in folds:
        fold.sort()
    return folds


def candidate_grid() -> list[tuple[str, dict[str, object]]]:
    return [
        (
            "baseline",
            {
                "iterations": 400,
                "depth": 6,
                "learning_rate": 0.05,
                "l2_leaf_reg": 3.0,
                "random_strength": 1.0,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 0.0,
            },
        ),
        (
            "d4_lr003_l230_b3",
            {
                "iterations": 500,
                "depth": 4,
                "learning_rate": 0.03,
                "l2_leaf_reg": 30.0,
                "random_strength": 3.0,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 1.0,
            },
        ),
        (
            "d4_lr002_l260_b6",
            {
                "iterations": 700,
                "depth": 4,
                "learning_rate": 0.02,
                "l2_leaf_reg": 60.0,
                "random_strength": 6.0,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 2.0,
            },
        ),
        (
            "d3_lr003_l280_b8",
            {
                "iterations": 800,
                "depth": 3,
                "learning_rate": 0.03,
                "l2_leaf_reg": 80.0,
                "random_strength": 8.0,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 3.0,
            },
        ),
        (
            "d3_lr005_l230_b5",
            {
                "iterations": 350,
                "depth": 3,
                "learning_rate": 0.05,
                "l2_leaf_reg": 30.0,
                "random_strength": 5.0,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 1.0,
            },
        ),
        (
            "d4_lr005_l220_b5",
            {
                "iterations": 250,
                "depth": 4,
                "learning_rate": 0.05,
                "l2_leaf_reg": 20.0,
                "random_strength": 5.0,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 1.0,
            },
        ),
        (
            "d5_lr003_l250_b3",
            {
                "iterations": 450,
                "depth": 5,
                "learning_rate": 0.03,
                "l2_leaf_reg": 50.0,
                "random_strength": 3.0,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 2.0,
            },
        ),
        (
            "d4_lr004_l210_mvs",
            {
                "iterations": 350,
                "depth": 4,
                "learning_rate": 0.04,
                "l2_leaf_reg": 10.0,
                "random_strength": 4.0,
                "bootstrap_type": "MVS",
                "subsample": 0.8,
            },
        ),
        (
            "d4_lr003_l230_bernoulli",
            {
                "iterations": 450,
                "depth": 4,
                "learning_rate": 0.03,
                "l2_leaf_reg": 30.0,
                "random_strength": 4.0,
                "bootstrap_type": "Bernoulli",
                "subsample": 0.7,
            },
        ),
        (
            "d3_lr004_l250_bernoulli",
            {
                "iterations": 500,
                "depth": 3,
                "learning_rate": 0.04,
                "l2_leaf_reg": 50.0,
                "random_strength": 6.0,
                "bootstrap_type": "Bernoulli",
                "subsample": 0.66,
            },
        ),
        (
            "d5_lr002_l280_b8",
            {
                "iterations": 700,
                "depth": 5,
                "learning_rate": 0.02,
                "l2_leaf_reg": 80.0,
                "random_strength": 8.0,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 3.0,
            },
        ),
        (
            "d4_lr006_l230_b3",
            {
                "iterations": 220,
                "depth": 4,
                "learning_rate": 0.06,
                "l2_leaf_reg": 30.0,
                "random_strength": 3.0,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 1.0,
            },
        ),
        (
            "d3_lr006_l220_b3",
            {
                "iterations": 220,
                "depth": 3,
                "learning_rate": 0.06,
                "l2_leaf_reg": 20.0,
                "random_strength": 3.0,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 1.0,
            },
        ),
        (
            "d5_lr004_l230_mvs",
            {
                "iterations": 300,
                "depth": 5,
                "learning_rate": 0.04,
                "l2_leaf_reg": 30.0,
                "random_strength": 5.0,
                "bootstrap_type": "MVS",
                "subsample": 0.8,
            },
        ),
        (
            "d4_lr003_l2100_b10",
            {
                "iterations": 500,
                "depth": 4,
                "learning_rate": 0.03,
                "l2_leaf_reg": 100.0,
                "random_strength": 10.0,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 4.0,
            },
        ),
        (
            "d3_lr002_l2120_b10",
            {
                "iterations": 900,
                "depth": 3,
                "learning_rate": 0.02,
                "l2_leaf_reg": 120.0,
                "random_strength": 10.0,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 4.0,
            },
        ),
    ]


def fit_catboost(
    x_train: list[list[float]],
    y_train: list[int],
    x_eval: list[list[float]],
    params: dict[str, object],
) -> list[float]:
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=THREAD_COUNT,
        **params,
    )
    model.fit(x_train, y_train)
    return [float(probability) for probability in model.predict_proba(x_eval)[:, 1]]


def evaluate_candidate(
    candidate_id: str,
    params: dict[str, object],
    matrix: list[list[float]],
    labels: list[int],
    folds: list[list[int]],
) -> CandidateResult:
    train_probabilities = fit_catboost(matrix, labels, matrix, params)
    train_auc = roc_auc_score(labels, train_probabilities)
    train_logloss = binary_logloss(labels, train_probabilities)

    oof_probabilities = [0.0] * len(labels)
    fold_aucs: list[float] = []
    for fold_indices in folds:
        validation_set = set(fold_indices)
        train_x = [row for index, row in enumerate(matrix) if index not in validation_set]
        train_y = [label for index, label in enumerate(labels) if index not in validation_set]
        valid_x = [matrix[index] for index in fold_indices]
        valid_y = [labels[index] for index in fold_indices]

        valid_probabilities = fit_catboost(train_x, train_y, valid_x, params)
        for index, probability in zip(fold_indices, valid_probabilities):
            oof_probabilities[index] = probability
        fold_aucs.append(roc_auc_score(valid_y, valid_probabilities))

    oof_auc = roc_auc_score(labels, oof_probabilities)
    oof_logloss = binary_logloss(labels, oof_probabilities)
    auc_gap = train_auc - oof_auc
    logloss_gap = oof_logloss - train_logloss
    balanced_score = oof_auc - 0.35 * auc_gap - 0.03 * logloss_gap - 0.02 * oof_logloss

    return CandidateResult(
        candidate_id=candidate_id,
        params=params,
        train_auc=train_auc,
        train_logloss=train_logloss,
        oof_auc=oof_auc,
        oof_logloss=oof_logloss,
        mean_fold_auc=sum(fold_aucs) / len(fold_aucs),
        fold_aucs=fold_aucs,
        auc_gap=auc_gap,
        logloss_gap=logloss_gap,
        balanced_score=balanced_score,
    )


def main() -> None:
    matrix, labels, feature_names = load_train_rows()
    folds = stratified_folds(labels, FOLDS, SEED)
    results = [
        evaluate_candidate(candidate_id, params, matrix, labels, folds)
        for candidate_id, params in candidate_grid()
    ]

    by_oof_auc = sorted(results, key=lambda row: (-row.oof_auc, row.oof_logloss, row.auc_gap))
    by_oof_logloss = sorted(results, key=lambda row: (row.oof_logloss, -row.oof_auc, row.auc_gap))
    by_balanced = sorted(results, key=lambda row: (-row.balanced_score, -row.oof_auc, row.oof_logloss))

    payload = {
        "experiment": "catboost_shared_cv_tuning",
        "train_dataset": str(TRAIN_PATH.relative_to(ROOT)),
        "rows": len(labels),
        "feature_names": feature_names,
        "folds": FOLDS,
        "seed": SEED,
        "thread_count": THREAD_COUNT,
        "best_by_oof_auc": asdict(by_oof_auc[0]),
        "best_by_oof_logloss": asdict(by_oof_logloss[0]),
        "best_by_balanced_score": asdict(by_balanced[0]),
        "results": [asdict(row) for row in by_balanced],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))

    print(f"saved {OUTPUT_PATH}")
    print("best_by_oof_auc", by_oof_auc[0].candidate_id, by_oof_auc[0].oof_auc, by_oof_auc[0].auc_gap)
    print(
        "best_by_oof_logloss",
        by_oof_logloss[0].candidate_id,
        by_oof_logloss[0].oof_logloss,
        by_oof_logloss[0].logloss_gap,
    )
    print(
        "best_by_balanced_score",
        by_balanced[0].candidate_id,
        by_balanced[0].oof_auc,
        by_balanced[0].oof_logloss,
        by_balanced[0].auc_gap,
        by_balanced[0].logloss_gap,
    )


if __name__ == "__main__":
    main()
