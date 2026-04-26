#!/usr/bin/env python3
"""Select an expanded meta-terminal library using shared-CV pairwise screening."""

from __future__ import annotations

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

from titanic_mle.metrics import binary_logloss, roc_auc_score


TRAIN_PATH = ROOT / "data" / "interim" / "titanic_unit_interval_train.csv"
PAIRWISE_PATH = ROOT / "data" / "processed" / "pairwise_top10_expressions_height_le_3_logloss.json"
CURRENT_STACKED_PATH = ROOT / "data" / "processed" / "meta_stacked_exact_search_top5_height_le_3.json"
CANDIDATE_OUTPUT_PATH = ROOT / "data" / "processed" / "pairwise_candidates_shared_cv.json"
LIBRARY_OUTPUT_PATH = ROOT / "data" / "processed" / "meta_terminal_library_top12_shared_cv.json"

SEED = 20_260_420
FOLDS = 5
TARGET_TERMINAL_COUNT = 12
SEX_EXTRA_SLOTS = 2


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    feature_a: str
    feature_b: str
    expression: str
    source_rank: int


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    feature_a: str
    feature_b: str
    expression: str
    source_rank: int
    train_auc: float
    train_logloss: float
    oof_auc: float
    oof_logloss: float
    mean_fold_auc: float
    fold_aucs: list[float]
    fitted_calibration_scale: float
    fitted_calibration_bias: float


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


def load_train_rows() -> tuple[list[dict[str, float]], list[int]]:
    with TRAIN_PATH.open() as handle:
        reader = csv.DictReader(handle)
        rows = []
        labels = []
        for row in reader:
            parsed = {key: float(value) for key, value in row.items() if key != "Survived"}
            rows.append(parsed)
            labels.append(int(row["Survived"]))
    return rows, labels


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


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def affine_logloss(scores: list[float], labels: list[int], scale: float, bias: float) -> float:
    probabilities = [min(max(sigmoid(scale * score + bias), 1e-15), 1 - 1e-15) for score in scores]
    return binary_logloss(labels, probabilities)


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


def apply_sigmoid_affine(scores: list[float], scale: float, bias: float) -> list[float]:
    return [min(max(sigmoid(scale * score + bias), 1e-15), 1 - 1e-15) for score in scores]


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


def abbreviate_feature(name: str) -> str:
    mapping = {
        "pclass_unit": "pc",
        "sex_unit": "sx",
        "age_unit": "ag",
        "age_missing": "agm",
        "sibsp_unit": "sb",
        "parch_unit": "pa",
        "fare_unit": "fr",
        "fare_missing": "frm",
        "embarked_unit": "em",
        "embarked_missing": "emm",
        "cabin_known": "cb",
        "family_size_unit": "fs",
        "is_alone": "ia",
    }
    return mapping.get(name, name.replace("_unit", "").replace("_", ""))


def load_pairwise_candidates() -> list[CandidateSpec]:
    pairwise_report = json.loads(PAIRWISE_PATH.read_text())
    candidates: list[CandidateSpec] = []
    for pair in pairwise_report["feature_pairs"]:
        feature_a = pair["feature_a"]
        feature_b = pair["feature_b"]
        for row in pair["top10"]:
            candidate_id = (
                f"{abbreviate_feature(feature_a)}_{abbreviate_feature(feature_b)}_"
                f"r{row['rank']}"
            )
            candidates.append(
                CandidateSpec(
                    candidate_id=candidate_id,
                    feature_a=feature_a,
                    feature_b=feature_b,
                    expression=row["expr"],
                    source_rank=int(row["rank"]),
                )
            )
    return candidates


def evaluate_candidate(
    spec: CandidateSpec,
    rows: list[dict[str, float]],
    labels: list[int],
    folds: list[list[int]],
) -> CandidateResult:
    expr = parse_pair_expr(spec.expression)
    scores = [expr.eval(row[spec.feature_a], row[spec.feature_b]) for row in rows]

    scale, bias = fit_sigmoid_affine(scores, labels)
    train_probabilities = apply_sigmoid_affine(scores, scale, bias)
    train_auc = roc_auc_score(labels, train_probabilities)
    train_logloss = binary_logloss(labels, train_probabilities)

    oof_probabilities = [0.0] * len(rows)
    fold_aucs: list[float] = []
    for fold_indices in folds:
        validation_set = set(fold_indices)
        train_indices = [index for index in range(len(rows)) if index not in validation_set]
        train_scores = [scores[index] for index in train_indices]
        train_labels = [labels[index] for index in train_indices]
        valid_scores = [scores[index] for index in fold_indices]
        valid_labels = [labels[index] for index in fold_indices]
        fold_scale, fold_bias = fit_sigmoid_affine(train_scores, train_labels)
        valid_probabilities = apply_sigmoid_affine(valid_scores, fold_scale, fold_bias)
        for index, probability in zip(fold_indices, valid_probabilities):
            oof_probabilities[index] = probability
        fold_aucs.append(roc_auc_score(valid_labels, valid_probabilities))

    return CandidateResult(
        candidate_id=spec.candidate_id,
        feature_a=spec.feature_a,
        feature_b=spec.feature_b,
        expression=spec.expression,
        source_rank=spec.source_rank,
        train_auc=train_auc,
        train_logloss=train_logloss,
        oof_auc=roc_auc_score(labels, oof_probabilities),
        oof_logloss=binary_logloss(labels, oof_probabilities),
        mean_fold_auc=sum(fold_aucs) / len(fold_aucs),
        fold_aucs=fold_aucs,
        fitted_calibration_scale=scale,
        fitted_calibration_bias=bias,
    )


def load_current_terminals() -> list[dict[str, object]]:
    stacked_report = json.loads(CURRENT_STACKED_PATH.read_text())
    terminals = []
    for row in stacked_report["terminals"]:
        terminals.append(
            {
                "id": row["id"],
                "feature_a": row["feature_a"],
                "feature_b": row["feature_b"],
                "source_expression": row["source_expression"],
                "source_report": row["source_report"],
                "selection_reason": "existing_meta_top5_anchor",
                "train_auc": row["calibrated_auc"],
                "train_logloss": row["calibrated_logloss"],
                "oof_auc": None,
                "oof_logloss": None,
            }
        )
    return terminals


def select_new_terminals(
    results: list[CandidateResult],
    current_pairs: set[tuple[str, str]],
    slots: int,
) -> list[dict[str, object]]:
    pair_to_results: dict[tuple[str, str], list[CandidateResult]] = {}
    for result in results:
        pair = (result.feature_a, result.feature_b)
        if pair in current_pairs:
            continue
        pair_to_results.setdefault(pair, []).append(result)

    sex_pairs: dict[tuple[str, str], CandidateResult] = {}
    non_sex_pairs: dict[tuple[str, str], list[CandidateResult]] = {}
    for pair, pair_results in pair_to_results.items():
        ordered = sorted(pair_results, key=lambda row: (row.oof_logloss, -row.oof_auc, row.source_rank))
        if "sex_unit" in pair:
            sex_pairs[pair] = ordered[0]
        else:
            non_sex_pairs[pair] = pair_results

    selected: list[tuple[CandidateResult, str]] = []
    used_pairs: set[tuple[str, str]] = set()

    for result in sorted(sex_pairs.values(), key=lambda row: (row.oof_logloss, -row.oof_auc))[:SEX_EXTRA_SLOTS]:
        selected.append((result, "best_new_sex_specialist"))
        used_pairs.add((result.feature_a, result.feature_b))

    logloss_rank: dict[tuple[str, str], int] = {}
    auc_rank: dict[tuple[str, str], int] = {}
    best_logloss_expr: dict[tuple[str, str], CandidateResult] = {}
    best_auc_expr: dict[tuple[str, str], CandidateResult] = {}

    non_sex_all = [row for pair_results in non_sex_pairs.values() for row in pair_results]
    for rank, row in enumerate(sorted(non_sex_all, key=lambda item: (item.oof_logloss, -item.oof_auc, item.source_rank))):
        pair = (row.feature_a, row.feature_b)
        logloss_rank.setdefault(pair, rank)
        best_logloss_expr.setdefault(pair, row)
    for rank, row in enumerate(sorted(non_sex_all, key=lambda item: (-item.oof_auc, item.oof_logloss, item.source_rank))):
        pair = (row.feature_a, row.feature_b)
        auc_rank.setdefault(pair, rank)
        best_auc_expr.setdefault(pair, row)

    pair_scores: list[tuple[int, int, tuple[str, str]]] = []
    for pair in non_sex_pairs:
        pair_scores.append((logloss_rank[pair] + auc_rank[pair], logloss_rank[pair], pair))
    pair_scores.sort()

    for _, _, pair in pair_scores:
        if pair in used_pairs:
            continue
        log_candidate = best_logloss_expr[pair]
        auc_candidate = best_auc_expr[pair]
        chosen = (
            auc_candidate
            if (-auc_candidate.oof_auc, auc_candidate.oof_logloss, auc_candidate.source_rank)
            < (-log_candidate.oof_auc, log_candidate.oof_logloss, log_candidate.source_rank)
            and auc_rank[pair] < logloss_rank[pair]
            else log_candidate
        )
        selected.append((chosen, "best_non_sex_pair_frontier"))
        used_pairs.add(pair)
        if len(selected) >= slots:
            break

    selected_rows: list[dict[str, object]] = []
    for result, reason in selected[:slots]:
        pair_has_sex = "sex_unit" in (result.feature_a, result.feature_b)
        selector = "sx" if pair_has_sex else "mx"
        selected_rows.append(
            {
                "id": f"{abbreviate_feature(result.feature_a)}_{abbreviate_feature(result.feature_b)}_{selector}",
                "feature_a": result.feature_a,
                "feature_b": result.feature_b,
                "source_expression": result.expression,
                "source_report": str(PAIRWISE_PATH.relative_to(ROOT)),
                "selection_reason": reason,
                "train_auc": result.train_auc,
                "train_logloss": result.train_logloss,
                "oof_auc": result.oof_auc,
                "oof_logloss": result.oof_logloss,
                "mean_fold_auc": result.mean_fold_auc,
                "source_rank": result.source_rank,
            }
        )

    return selected_rows[:slots]


def main() -> None:
    rows, labels = load_train_rows()
    folds = stratified_folds(labels, FOLDS, SEED)

    candidates = load_pairwise_candidates()
    results = [evaluate_candidate(spec, rows, labels, folds) for spec in candidates]
    results.sort(key=lambda row: (row.oof_logloss, -row.oof_auc, row.source_rank))

    current_terminals = load_current_terminals()
    current_pairs = {(row["feature_a"], row["feature_b"]) for row in current_terminals}
    new_slots = TARGET_TERMINAL_COUNT - len(current_terminals)
    selected_new_terminals = select_new_terminals(results, current_pairs, new_slots)

    candidate_payload = {
        "experiment": "pairwise_candidates_shared_cv",
        "source_pairwise_report": str(PAIRWISE_PATH.relative_to(ROOT)),
        "train_dataset": str(TRAIN_PATH.relative_to(ROOT)),
        "folds": FOLDS,
        "seed": SEED,
        "candidate_count": len(results),
        "overall_top20_by_oof_logloss": [row.__dict__ for row in results[:20]],
        "overall_top20_by_oof_auc": [
            row.__dict__
            for row in sorted(results, key=lambda row: (-row.oof_auc, row.oof_logloss, row.source_rank))[:20]
        ],
        "results": [row.__dict__ for row in results],
    }
    CANDIDATE_OUTPUT_PATH.write_text(json.dumps(candidate_payload, indent=2))

    library_payload = {
        "experiment": "meta_terminal_library_shared_cv_selection",
        "source_pairwise_report": str(PAIRWISE_PATH.relative_to(ROOT)),
        "source_stacked_report": str(CURRENT_STACKED_PATH.relative_to(ROOT)),
        "train_dataset": str(TRAIN_PATH.relative_to(ROOT)),
        "folds": FOLDS,
        "seed": SEED,
        "target_terminal_count": TARGET_TERMINAL_COUNT,
        "current_terminals": current_terminals,
        "selected_new_terminals": selected_new_terminals,
        "terminal_library": current_terminals + selected_new_terminals,
    }
    LIBRARY_OUTPUT_PATH.write_text(json.dumps(library_payload, indent=2))

    print(f"saved {CANDIDATE_OUTPUT_PATH}")
    print(f"saved {LIBRARY_OUTPUT_PATH}")
    print("selected new terminals:")
    for row in selected_new_terminals:
        print(
            f"{row['id']}: {row['feature_a']} x {row['feature_b']} "
            f"oof_logloss={row['oof_logloss']:.16f} "
            f"oof_auc={row['oof_auc']:.16f} "
            f"expr={row['source_expression']}"
        )


if __name__ == "__main__":
    main()
