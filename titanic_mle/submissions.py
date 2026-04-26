"""Helpers for preparing Kaggle Titanic submissions."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from catboost import CatBoostClassifier

from .metrics import binary_logloss, roc_auc_score, sigmoid


THREAD_COUNT = 8


@dataclass(frozen=True)
class TerminalSpec:
    terminal_id: str
    feature_a: str
    feature_b: str
    pair_expr: str
    calibration_scale: float
    calibration_bias: float


@dataclass(frozen=True)
class EmlModelSpec:
    model_name: str
    source_report: str
    meta_expr: str
    terminals: list[TerminalSpec]
    top_calibration_scale: float
    top_calibration_bias: float


@dataclass(frozen=True)
class FamilySubmissionResult:
    family: str
    candidate_id: str
    source_report: str | None
    threshold: float
    oof_accuracy: float
    train_accuracy: float
    train_auc: float
    train_logloss: float
    oof_auc: float
    oof_logloss: float
    test_positive_count: int
    submission_path: str


def load_unit_interval_rows(
    path: Path,
) -> tuple[list[dict[str, float]], list[int] | None, list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, float]] = []
        labels: list[int] | None = []
        feature_names: list[str] | None = None
        for row in reader:
            if feature_names is None:
                feature_names = [
                    name for name in row.keys() if name not in {"PassengerId", "Survived"}
                ]
            parsed = {key: float(value) for key, value in row.items() if key != "Survived"}
            rows.append(parsed)
            if "Survived" in row:
                assert labels is not None
                labels.append(int(row["Survived"]))
            else:
                labels = None
        assert feature_names is not None
    return rows, labels, feature_names


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


def accuracy_score(labels: list[int], predictions: list[int]) -> float:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    if not labels:
        raise ValueError("inputs must be non-empty")
    correct = sum(int(label == prediction) for label, prediction in zip(labels, predictions))
    return correct / len(labels)


def threshold_predictions(probabilities: list[float], threshold: float) -> list[int]:
    return [1 if probability >= threshold else 0 for probability in probabilities]


def best_threshold_by_accuracy(probabilities: list[float], labels: list[int]) -> tuple[float, float]:
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    if not probabilities:
        raise ValueError("inputs must be non-empty")

    unique_values = sorted(set(probabilities))
    candidates = [0.0, 0.5, 1.0]
    candidates.extend((left + right) / 2.0 for left, right in zip(unique_values, unique_values[1:]))
    candidates = sorted(set(candidates))

    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in candidates:
        accuracy = accuracy_score(labels, threshold_predictions(probabilities, threshold))
        if accuracy > best_accuracy + 1e-12:
            best_accuracy = accuracy
            best_threshold = threshold
            continue
        if abs(accuracy - best_accuracy) <= 1e-12 and abs(threshold - 0.5) < abs(best_threshold - 0.5):
            best_threshold = threshold

    return best_threshold, best_accuracy


def build_matrix(rows: list[dict[str, float]], feature_names: list[str]) -> list[list[float]]:
    return [[row[name] for name in feature_names] for row in rows]


def fit_catboost_probabilities(
    train_x: list[list[float]],
    train_y: list[int],
    eval_x: list[list[float]],
    params: dict[str, object],
    seed: int,
) -> list[float]:
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=THREAD_COUNT,
        **params,
    )
    model.fit(train_x, train_y)
    return [float(probability) for probability in model.predict_proba(eval_x)[:, 1]]


def catboost_oof_probabilities(
    matrix: list[list[float]],
    labels: list[int],
    folds: list[list[int]],
    params: dict[str, object],
    seed: int,
) -> list[float]:
    oof_probabilities = [0.0] * len(labels)
    for fold_indices in folds:
        validation_set = set(fold_indices)
        train_x = [row for index, row in enumerate(matrix) if index not in validation_set]
        train_y = [label for index, label in enumerate(labels) if index not in validation_set]
        valid_x = [matrix[index] for index in fold_indices]
        valid_probabilities = fit_catboost_probabilities(train_x, train_y, valid_x, params, seed)
        for index, probability in zip(fold_indices, valid_probabilities):
            oof_probabilities[index] = probability
    return oof_probabilities


class PairExpr:
    def __init__(self, kind: str, left: "PairExpr | None" = None, right: "PairExpr | None" = None):
        self.kind = kind
        self.left = left
        self.right = right

    def eval(self, x0: float, x1: float) -> float:
        if self.kind == "1":
            return 1.0
        if self.kind == "x0":
            return x0
        if self.kind == "x1":
            return x1
        left_value = self.left.eval(x0, x1)
        right_value = self.right.eval(x0, x1)
        if right_value <= 0.0:
            raise ValueError("log domain")
        value = math.exp(left_value) - math.log(right_value)
        if not math.isfinite(value):
            raise ValueError("non-finite")
        return value


class MetaExpr:
    def __init__(self, kind: str, left: "MetaExpr | None" = None, right: "MetaExpr | None" = None):
        self.kind = kind
        self.left = left
        self.right = right

    def eval(self, terminals: dict[str, float]) -> float:
        if self.kind != "node":
            return terminals[self.kind]
        left_value = self.left.eval(terminals)
        right_value = self.right.eval(terminals)
        if right_value <= 0.0:
            raise ValueError("log domain")
        value = math.exp(left_value) - math.log(right_value)
        if not math.isfinite(value):
            raise ValueError("non-finite")
        return value


def parse_pair_expr(text: str) -> PairExpr:
    index = 0

    def skip_ws() -> None:
        nonlocal index
        while index < len(text) and text[index].isspace():
            index += 1

    def parse() -> PairExpr:
        nonlocal index
        skip_ws()
        if text.startswith("x0", index):
            index += 2
            return PairExpr("x0")
        if text.startswith("x1", index):
            index += 2
            return PairExpr("x1")
        if text[index] == "1":
            index += 1
            return PairExpr("1")
        if text[index] != "(":
            raise ValueError(f"unexpected token in pair expr: {text[index:]}")
        index += 1
        left = parse()
        right = parse()
        skip_ws()
        if text[index] != ")":
            raise ValueError("missing closing )")
        index += 1
        return PairExpr("node", left, right)

    expr = parse()
    skip_ws()
    if index != len(text):
        raise ValueError("trailing input in pair expr")
    return expr


def tokenize_meta(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for ch in text:
        if ch in "()":
            if current:
                tokens.append("".join(current))
                current.clear()
            tokens.append(ch)
        elif ch.isspace():
            if current:
                tokens.append("".join(current))
                current.clear()
        else:
            current.append(ch)
    if current:
        tokens.append("".join(current))
    return tokens


def parse_meta_expr(text: str) -> MetaExpr:
    tokens = tokenize_meta(text)
    index = 0

    def parse() -> MetaExpr:
        nonlocal index
        token = tokens[index]
        if token == "(":
            index += 1
            left = parse()
            right = parse()
            if tokens[index] != ")":
                raise ValueError("missing closing ) in meta expr")
            index += 1
            return MetaExpr("node", left, right)
        index += 1
        return MetaExpr(token)

    expr = parse()
    if index != len(tokens):
        raise ValueError("trailing tokens in meta expr")
    return expr


def fit_sigmoid_affine(scores: list[float], labels: list[int]) -> tuple[float, float]:
    scale = 1.0
    bias = 0.0
    best_loss = affine_logloss(scores, labels, scale, bias)

    for _ in range(25):
        g_scale = 0.0
        g_bias = 0.0
        h_ss = 0.0
        h_sb = 0.0
        h_bb = 0.0
        for score, label in zip(scores, labels):
            probability = sigmoid(scale * score + bias)
            error = probability - float(label)
            variance = probability * (1.0 - probability)
            g_scale += error * score
            g_bias += error
            h_ss += variance * score * score
            h_sb += variance * score
            h_bb += variance

        determinant = h_ss * h_bb - h_sb * h_sb
        if determinant <= 1e-18:
            break

        step_scale = (h_bb * g_scale - h_sb * g_bias) / determinant
        step_bias = (-h_sb * g_scale + h_ss * g_bias) / determinant

        step_factor = 1.0
        improved: tuple[float, float, float] | None = None
        while step_factor > 1e-8:
            candidate_scale = scale - step_factor * step_scale
            candidate_bias = bias - step_factor * step_bias
            if candidate_scale <= 1e-10:
                step_factor *= 0.5
                continue
            candidate_loss = affine_logloss(scores, labels, candidate_scale, candidate_bias)
            if candidate_loss < best_loss - 1e-12:
                improved = (candidate_scale, candidate_bias, candidate_loss)
                break
            step_factor *= 0.5

        if improved is None:
            break

        scale, bias, best_loss = improved
        if abs(step_factor * step_scale) < 1e-10 and abs(step_factor * step_bias) < 1e-10:
            break

    return scale, bias


def affine_logloss(scores: list[float], labels: list[int], scale: float, bias: float) -> float:
    probabilities = apply_sigmoid_affine(scores, scale, bias)
    return binary_logloss(labels, probabilities)


def apply_sigmoid_affine(scores: list[float], scale: float, bias: float) -> list[float]:
    return [min(max(sigmoid(scale * score + bias), 1e-15), 1.0 - 1e-15) for score in scores]


def load_eml_model(report_path: Path, model_name: str) -> EmlModelSpec:
    report = json.loads(report_path.read_text())
    if "best_by_auc" in report:
        branch = report["best_by_auc"] if model_name.endswith("_auc") else report["best_by_logloss"]
        meta_expr = branch["expr"]
        top_scale = float(branch["calibration_scale"])
        top_bias = float(branch["calibration_bias"])
    else:
        meta_expr = report["best_expression"]
        top_scale = float(report["best_calibration_scale"])
        top_bias = float(report["best_calibration_bias"])

    terminals = [
        TerminalSpec(
            terminal_id=row["id"],
            feature_a=row["feature_a"],
            feature_b=row["feature_b"],
            pair_expr=row["source_expression"],
            calibration_scale=float(row["fitted_calibration_scale"]),
            calibration_bias=float(row["fitted_calibration_bias"]),
        )
        for row in report["terminals"]
    ]

    return EmlModelSpec(
        model_name=model_name,
        source_report=str(report_path),
        meta_expr=meta_expr,
        terminals=terminals,
        top_calibration_scale=top_scale,
        top_calibration_bias=top_bias,
    )


def eml_train_test_probabilities(
    spec: EmlModelSpec,
    train_rows: list[dict[str, float]],
    test_rows: list[dict[str, float]],
) -> tuple[list[float], list[float]]:
    pair_exprs = {terminal.terminal_id: parse_pair_expr(terminal.pair_expr) for terminal in spec.terminals}
    meta_expr = parse_meta_expr(spec.meta_expr)

    train_terminal_probabilities: dict[str, list[float]] = {}
    test_terminal_probabilities: dict[str, list[float]] = {}

    for terminal in spec.terminals:
        train_scores = [
            pair_exprs[terminal.terminal_id].eval(row[terminal.feature_a], row[terminal.feature_b])
            for row in train_rows
        ]
        test_scores = [
            pair_exprs[terminal.terminal_id].eval(row[terminal.feature_a], row[terminal.feature_b])
            for row in test_rows
        ]
        train_terminal_probabilities[terminal.terminal_id] = apply_sigmoid_affine(
            train_scores,
            terminal.calibration_scale,
            terminal.calibration_bias,
        )
        test_terminal_probabilities[terminal.terminal_id] = apply_sigmoid_affine(
            test_scores,
            terminal.calibration_scale,
            terminal.calibration_bias,
        )

    train_meta_scores = [
        meta_expr.eval({terminal_id: values[row_index] for terminal_id, values in train_terminal_probabilities.items()})
        for row_index in range(len(train_rows))
    ]
    test_meta_scores = [
        meta_expr.eval({terminal_id: values[row_index] for terminal_id, values in test_terminal_probabilities.items()})
        for row_index in range(len(test_rows))
    ]

    train_probabilities = apply_sigmoid_affine(
        train_meta_scores,
        spec.top_calibration_scale,
        spec.top_calibration_bias,
    )
    test_probabilities = apply_sigmoid_affine(
        test_meta_scores,
        spec.top_calibration_scale,
        spec.top_calibration_bias,
    )
    return train_probabilities, test_probabilities


def eml_oof_probabilities(
    spec: EmlModelSpec,
    rows: list[dict[str, float]],
    labels: list[int],
    folds: list[list[int]],
) -> list[float]:
    pair_exprs = {terminal.terminal_id: parse_pair_expr(terminal.pair_expr) for terminal in spec.terminals}
    meta_expr = parse_meta_expr(spec.meta_expr)

    terminal_probabilities: dict[str, list[float]] = {}
    for terminal in spec.terminals:
        scores = [
            pair_exprs[terminal.terminal_id].eval(row[terminal.feature_a], row[terminal.feature_b])
            for row in rows
        ]
        terminal_probabilities[terminal.terminal_id] = apply_sigmoid_affine(
            scores,
            terminal.calibration_scale,
            terminal.calibration_bias,
        )

    oof_probabilities = [0.0] * len(rows)
    for fold_indices in folds:
        validation_set = set(fold_indices)
        train_indices = [index for index in range(len(rows)) if index not in validation_set]
        train_labels = [labels[index] for index in train_indices]
        train_meta_scores = [
            meta_expr.eval({terminal_id: values[index] for terminal_id, values in terminal_probabilities.items()})
            for index in train_indices
        ]
        valid_meta_scores = [
            meta_expr.eval({terminal_id: values[index] for terminal_id, values in terminal_probabilities.items()})
            for index in fold_indices
        ]
        meta_scale, meta_bias = fit_sigmoid_affine(train_meta_scores, train_labels)
        valid_probabilities = apply_sigmoid_affine(valid_meta_scores, meta_scale, meta_bias)
        for index, probability in zip(fold_indices, valid_probabilities):
            oof_probabilities[index] = probability

    return oof_probabilities


def write_submission_csv(path: Path, passenger_ids: list[int | str], predictions: list[int]) -> None:
    if len(passenger_ids) != len(predictions):
        raise ValueError("passenger_ids and predictions must have the same length")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["PassengerId", "Survived"])
        for passenger_id, prediction in zip(passenger_ids, predictions):
            writer.writerow([passenger_id, prediction])
