#!/usr/bin/env python3
"""Shared CV benchmark for stacked EML frontiers and CatBoost."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catboost import CatBoostClassifier

from titanic_mle.catboost_profiles import (
    CATBOOST_BASELINE_PARAMS,
    CATBOOST_BASELINE_PROFILE_ID,
    CATBOOST_TUNED_BEST_AUC_PARAMS,
    CATBOOST_TUNED_BEST_AUC_PROFILE_ID,
    CATBOOST_TUNED_BALANCED_PARAMS,
    CATBOOST_TUNED_BALANCED_PROFILE_ID,
)
from titanic_mle.metrics import binary_logloss, roc_auc_score


TRAIN_PATH = ROOT / "data" / "interim" / "titanic_unit_interval_train.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "eml_catboost_shared_cv_benchmark.json"

EML_REPORTS = {
    "exact_top5_auc": ROOT / "data" / "processed" / "meta_stacked_exact_search_top5_height_le_3.json",
    "meta_ga_auc_iter3": ROOT / "data" / "processed" / "ga_meta_stacked_top5__iter3__auc.json",
    "meta_ga_calibrated_logloss_iter4": ROOT
    / "data"
    / "processed"
    / "ga_meta_stacked_top5__iter4__calibrated_logloss.json",
}


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


@dataclass(frozen=True)
class CvResult:
    model_name: str
    train_auc: float
    train_logloss: float
    oof_auc: float
    oof_logloss: float
    mean_fold_auc: float
    fold_aucs: list[float]
    source_report: str | None = None
    meta_expr: str | None = None
    feature_names: list[str] | None = None
    params: dict[str, object] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20_260_420)
    parser.add_argument(
        "--eml-report",
        action="append",
        default=[],
        help="Benchmark an additional EML report in the form model_name=path/to/report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Where to save the combined benchmark JSON",
    )
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


def parse_eml_reports(report_specs: list[str]) -> list[tuple[str, Path]]:
    if not report_specs:
        return [(name, path) for name, path in EML_REPORTS.items()]

    parsed: list[tuple[str, Path]] = []
    for spec in report_specs:
        if "=" not in spec:
            raise ValueError(f"invalid --eml-report spec: {spec}")
        model_name, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        parsed.append((model_name, path))
    return parsed


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


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


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
            target = float(label)
            error = probability - target
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
    probabilities = [min(max(sigmoid(scale * score + bias), 1e-15), 1 - 1e-15) for score in scores]
    return binary_logloss(labels, probabilities)


def apply_sigmoid_affine(scores: list[float], scale: float, bias: float) -> list[float]:
    return [min(max(sigmoid(scale * score + bias), 1e-15), 1 - 1e-15) for score in scores]


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
        if self.kind not in {"node"}:
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
        raise ValueError("trailing input")
    return expr


def tokenize_meta(text: str) -> list[str]:
    tokens: list[str] = []
    current = []
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
                raise ValueError("missing closing )")
            index += 1
            return MetaExpr("node", left, right)
        index += 1
        return MetaExpr(token)

    expr = parse()
    if index != len(tokens):
        raise ValueError("trailing input")
    return expr


def load_eml_model(report_path: Path, model_name: str) -> EmlModelSpec:
    report = json.loads(report_path.read_text())
    if "best_by_auc" in report:
        meta_expr = report["best_by_auc"]["expr"] if model_name.endswith("_auc") else report["best_by_logloss"]["expr"]
        terminals = [
            TerminalSpec(
                terminal_id=row["id"],
                feature_a=row["feature_a"],
                feature_b=row["feature_b"],
                pair_expr=row["source_expression"],
                calibration_scale=row["fitted_calibration_scale"],
                calibration_bias=row["fitted_calibration_bias"],
            )
            for row in report["terminals"]
        ]
    else:
        meta_expr = report["best_expression"]
        terminals = [
            TerminalSpec(
                terminal_id=row["id"],
                feature_a=row["feature_a"],
                feature_b=row["feature_b"],
                pair_expr=row["source_expression"],
                calibration_scale=row["fitted_calibration_scale"],
                calibration_bias=row["fitted_calibration_bias"],
            )
            for row in report["terminals"]
        ]
    return EmlModelSpec(
        model_name=model_name,
        source_report=str(report_path.relative_to(ROOT)),
        meta_expr=meta_expr,
        terminals=terminals,
    )


def benchmark_eml_model(
    spec: EmlModelSpec,
    rows: list[dict[str, float]],
    labels: list[int],
    folds: list[list[int]],
) -> CvResult:
    pair_exprs = {terminal.terminal_id: parse_pair_expr(terminal.pair_expr) for terminal in spec.terminals}
    meta_expr = parse_meta_expr(spec.meta_expr)

    raw_terminal_scores = {
        terminal.terminal_id: [
            pair_exprs[terminal.terminal_id].eval(row[terminal.feature_a], row[terminal.feature_b])
            for row in rows
        ]
        for terminal in spec.terminals
    }

    full_terminal_probabilities: dict[str, list[float]] = {}
    for terminal in spec.terminals:
        scores = raw_terminal_scores[terminal.terminal_id]
        full_terminal_probabilities[terminal.terminal_id] = apply_sigmoid_affine(
            scores,
            terminal.calibration_scale,
            terminal.calibration_bias,
        )

    full_meta_scores = [
        meta_expr.eval({terminal_id: values[row_index] for terminal_id, values in full_terminal_probabilities.items()})
        for row_index in range(len(rows))
    ]
    full_scale, full_bias = fit_sigmoid_affine(full_meta_scores, labels)
    full_probabilities = apply_sigmoid_affine(full_meta_scores, full_scale, full_bias)
    train_auc = roc_auc_score(labels, full_probabilities)
    train_logloss = binary_logloss(labels, full_probabilities)

    oof_probabilities = [0.0] * len(rows)
    fold_aucs: list[float] = []

    for fold_indices in folds:
        validation_set = set(fold_indices)
        train_indices = [index for index in range(len(rows)) if index not in validation_set]
        train_labels = [labels[index] for index in train_indices]
        valid_labels = [labels[index] for index in fold_indices]

        train_meta_scores = [
            meta_expr.eval(
                {
                    terminal_id: values[index]
                    for terminal_id, values in full_terminal_probabilities.items()
                }
            )
            for index in train_indices
        ]
        valid_meta_scores = [
            meta_expr.eval(
                {
                    terminal_id: values[index]
                    for terminal_id, values in full_terminal_probabilities.items()
                }
            )
            for index in fold_indices
        ]
        meta_scale, meta_bias = fit_sigmoid_affine(train_meta_scores, train_labels)
        valid_probabilities = apply_sigmoid_affine(valid_meta_scores, meta_scale, meta_bias)
        for index, probability in zip(fold_indices, valid_probabilities):
            oof_probabilities[index] = probability
        fold_aucs.append(roc_auc_score(valid_labels, valid_probabilities))

    oof_auc = roc_auc_score(labels, oof_probabilities)
    oof_logloss = binary_logloss(labels, oof_probabilities)

    return CvResult(
        model_name=spec.model_name,
        train_auc=train_auc,
        train_logloss=train_logloss,
        oof_auc=oof_auc,
        oof_logloss=oof_logloss,
        mean_fold_auc=sum(fold_aucs) / len(fold_aucs),
        fold_aucs=fold_aucs,
        source_report=spec.source_report,
        meta_expr=spec.meta_expr,
    )


def build_matrix(rows: list[dict[str, float]], feature_names: list[str]) -> list[list[float]]:
    return [[row[name] for name in feature_names] for row in rows]


def fit_catboost(
    x_train: list[list[float]],
    y_train: list[int],
    x_eval: list[list[float]],
    seed: int,
    params: dict[str, object],
) -> list[float]:
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        **params,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_eval)[:, 1]
    return [float(probability) for probability in probabilities]


def benchmark_catboost_model(
    model_name: str,
    feature_names: list[str],
    rows: list[dict[str, float]],
    labels: list[int],
    folds: list[list[int]],
    seed: int,
    params: dict[str, object],
) -> CvResult:
    matrix = build_matrix(rows, feature_names)
    train_probabilities = fit_catboost(matrix, labels, matrix, seed, params)
    train_auc = roc_auc_score(labels, train_probabilities)
    train_logloss = binary_logloss(labels, train_probabilities)

    oof_probabilities = [0.0] * len(rows)
    fold_aucs: list[float] = []
    for fold_indices in folds:
        validation_set = set(fold_indices)
        train_x = [row for index, row in enumerate(matrix) if index not in validation_set]
        train_y = [label for index, label in enumerate(labels) if index not in validation_set]
        valid_x = [matrix[index] for index in fold_indices]
        valid_y = [labels[index] for index in fold_indices]
        valid_probabilities = fit_catboost(train_x, train_y, valid_x, seed, params)
        for index, probability in zip(fold_indices, valid_probabilities):
            oof_probabilities[index] = probability
        fold_aucs.append(roc_auc_score(valid_y, valid_probabilities))

    oof_auc = roc_auc_score(labels, oof_probabilities)
    oof_logloss = binary_logloss(labels, oof_probabilities)
    return CvResult(
        model_name=model_name,
        train_auc=train_auc,
        train_logloss=train_logloss,
        oof_auc=oof_auc,
        oof_logloss=oof_logloss,
        mean_fold_auc=sum(fold_aucs) / len(fold_aucs),
        fold_aucs=fold_aucs,
        feature_names=feature_names,
        params=params,
    )


def main() -> None:
    args = parse_args()
    rows, labels, all_unit_features = load_train_rows()
    folds = stratified_folds(labels, args.folds, args.seed)

    eml_results = [
        benchmark_eml_model(load_eml_model(report_path, model_name), rows, labels, folds)
        for model_name, report_path in parse_eml_reports(args.eml_report)
    ]

    catboost_results = [
        benchmark_catboost_model(
            f"catboost_all_unit_features_{CATBOOST_BASELINE_PROFILE_ID}",
            all_unit_features,
            rows,
            labels,
            folds,
            args.seed,
            CATBOOST_BASELINE_PARAMS,
        ),
        benchmark_catboost_model(
            f"catboost_all_unit_features_{CATBOOST_TUNED_BALANCED_PROFILE_ID}",
            all_unit_features,
            rows,
            labels,
            folds,
            args.seed,
            CATBOOST_TUNED_BALANCED_PARAMS,
        ),
        benchmark_catboost_model(
            f"catboost_all_unit_features_{CATBOOST_TUNED_BEST_AUC_PROFILE_ID}",
            all_unit_features,
            rows,
            labels,
            folds,
            args.seed,
            CATBOOST_TUNED_BEST_AUC_PARAMS,
        ),
    ]

    payload = {
        "train_dataset": str(TRAIN_PATH.relative_to(ROOT)),
        "rows": len(rows),
        "folds": args.folds,
        "seed": args.seed,
        "catboost_profiles": {
            CATBOOST_BASELINE_PROFILE_ID: CATBOOST_BASELINE_PARAMS,
            CATBOOST_TUNED_BALANCED_PROFILE_ID: CATBOOST_TUNED_BALANCED_PARAMS,
            CATBOOST_TUNED_BEST_AUC_PROFILE_ID: CATBOOST_TUNED_BEST_AUC_PARAMS,
        },
        "eml_models": [result.__dict__ for result in eml_results],
        "catboost_models": [result.__dict__ for result in catboost_results],
    }
    args.output.write_text(json.dumps(payload, indent=2))

    print(f"saved {args.output}")
    for result in eml_results + catboost_results:
        print(
            f"{result.model_name}: "
            f"train_auc={result.train_auc:.16f} "
            f"oof_auc={result.oof_auc:.16f} "
            f"train_logloss={result.train_logloss:.16f} "
            f"oof_logloss={result.oof_logloss:.16f}"
        )


if __name__ == "__main__":
    main()
