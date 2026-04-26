#!/usr/bin/env python3
"""Prepare Kaggle-ready Titanic submissions for CatBoost and EML."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from titanic_mle.catboost_profiles import (
    CATBOOST_BASELINE_PARAMS,
    CATBOOST_BASELINE_PROFILE_ID,
    CATBOOST_TUNED_BEST_AUC_PARAMS,
    CATBOOST_TUNED_BEST_AUC_PROFILE_ID,
    CATBOOST_TUNED_BEST_LOGLOSS_PARAMS,
    CATBOOST_TUNED_BEST_LOGLOSS_PROFILE_ID,
    CATBOOST_TUNED_BALANCED_PARAMS,
    CATBOOST_TUNED_BALANCED_PROFILE_ID,
)
from titanic_mle.metrics import binary_logloss, roc_auc_score
from titanic_mle.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, SUBMISSIONS_DATA_DIR
from titanic_mle.submissions import (
    FamilySubmissionResult,
    accuracy_score,
    best_threshold_by_accuracy,
    build_matrix,
    catboost_oof_probabilities,
    eml_oof_probabilities,
    eml_train_test_probabilities,
    fit_catboost_probabilities,
    load_eml_model,
    load_unit_interval_rows,
    stratified_folds,
    threshold_predictions,
    write_submission_csv,
)


SEED = 20_260_420
FOLDS = 5
TRAIN_PATH = INTERIM_DATA_DIR / "titanic_unit_interval_train.csv"
TEST_PATH = INTERIM_DATA_DIR / "titanic_unit_interval_test.csv"
SUMMARY_PATH = SUBMISSIONS_DATA_DIR / "submission_summary.json"

CATBOOST_CANDIDATES = [
    (CATBOOST_BASELINE_PROFILE_ID, CATBOOST_BASELINE_PARAMS),
    (CATBOOST_TUNED_BEST_AUC_PROFILE_ID, CATBOOST_TUNED_BEST_AUC_PARAMS),
    (CATBOOST_TUNED_BALANCED_PROFILE_ID, CATBOOST_TUNED_BALANCED_PARAMS),
    (CATBOOST_TUNED_BEST_LOGLOSS_PROFILE_ID, CATBOOST_TUNED_BEST_LOGLOSS_PARAMS),
]

EML_CANDIDATE_REPORTS = [
    ("exact_top5_auc", PROCESSED_DATA_DIR / "meta_stacked_exact_search_top5_height_le_3.json"),
    ("meta_ga_auc_iter3", PROCESSED_DATA_DIR / "ga_meta_stacked_top5__iter3__auc.json"),
    ("meta_ga_calibrated_logloss_iter4", PROCESSED_DATA_DIR / "ga_meta_stacked_top5__iter4__calibrated_logloss.json"),
    ("meta_ga_top12_auc_iter1", PROCESSED_DATA_DIR / "ga_meta_stacked_top12__lib12_iter1__auc_calibrated_logloss.json"),
    (
        "meta_ga_top12_calibrated_logloss_iter2",
        PROCESSED_DATA_DIR / "ga_meta_stacked_top12__lib12_iter2__calibrated_logloss.json",
    ),
]


def evaluate_catboost_family(
    train_rows: list[dict[str, float]],
    labels: list[int],
    test_rows: list[dict[str, float]],
    feature_names: list[str],
    folds: list[list[int]],
) -> tuple[FamilySubmissionResult, list[FamilySubmissionResult]]:
    train_matrix = build_matrix(train_rows, feature_names)
    test_matrix = build_matrix(test_rows, feature_names)
    passenger_ids = [int(row["PassengerId"]) for row in test_rows]

    results: list[FamilySubmissionResult] = []
    for candidate_id, params in CATBOOST_CANDIDATES:
        train_probabilities = fit_catboost_probabilities(train_matrix, labels, train_matrix, params, SEED)
        oof_probabilities = catboost_oof_probabilities(train_matrix, labels, folds, params, SEED)
        test_probabilities = fit_catboost_probabilities(train_matrix, labels, test_matrix, params, SEED)

        threshold, oof_accuracy = best_threshold_by_accuracy(oof_probabilities, labels)
        train_predictions = threshold_predictions(train_probabilities, threshold)
        train_accuracy = accuracy_score(labels, train_predictions)
        test_predictions = threshold_predictions(test_probabilities, threshold)

        submission_path = SUBMISSIONS_DATA_DIR / f"catboost_{candidate_id}_submission.csv"
        write_submission_csv(submission_path, passenger_ids, test_predictions)

        result = FamilySubmissionResult(
            family="catboost",
            candidate_id=candidate_id,
            source_report=None,
            threshold=threshold,
            oof_accuracy=oof_accuracy,
            train_accuracy=train_accuracy,
            train_auc=roc_auc_score(labels, train_probabilities),
            train_logloss=binary_logloss(labels, train_probabilities),
            oof_auc=roc_auc_score(labels, oof_probabilities),
            oof_logloss=binary_logloss(labels, oof_probabilities),
            test_positive_count=sum(test_predictions),
            submission_path=str(submission_path),
        )
        results.append(result)

    results.sort(
        key=lambda row: (
            -row.oof_accuracy,
            -row.oof_auc,
            row.oof_logloss,
            abs(row.threshold - 0.5),
        )
    )
    return results[0], results


def evaluate_eml_family(
    train_rows: list[dict[str, float]],
    labels: list[int],
    test_rows: list[dict[str, float]],
    folds: list[list[int]],
) -> tuple[FamilySubmissionResult, list[FamilySubmissionResult], list[dict[str, str]]]:
    passenger_ids = [int(row["PassengerId"]) for row in test_rows]

    results: list[FamilySubmissionResult] = []
    skipped: list[dict[str, str]] = []
    for candidate_id, report_path in EML_CANDIDATE_REPORTS:
        spec = load_eml_model(report_path, candidate_id)
        try:
            train_probabilities, test_probabilities = eml_train_test_probabilities(spec, train_rows, test_rows)
        except ValueError as exc:
            skipped.append(
                {
                    "candidate_id": candidate_id,
                    "source_report": str(report_path),
                    "reason": str(exc),
                }
            )
            continue
        oof_probabilities = eml_oof_probabilities(spec, train_rows, labels, folds)

        threshold, oof_accuracy = best_threshold_by_accuracy(oof_probabilities, labels)
        train_predictions = threshold_predictions(train_probabilities, threshold)
        train_accuracy = accuracy_score(labels, train_predictions)
        test_predictions = threshold_predictions(test_probabilities, threshold)

        submission_path = SUBMISSIONS_DATA_DIR / f"eml_{candidate_id}_submission.csv"
        write_submission_csv(submission_path, passenger_ids, test_predictions)

        results.append(
            FamilySubmissionResult(
                family="eml",
                candidate_id=candidate_id,
                source_report=str(report_path),
                threshold=threshold,
                oof_accuracy=oof_accuracy,
                train_accuracy=train_accuracy,
                train_auc=roc_auc_score(labels, train_probabilities),
                train_logloss=binary_logloss(labels, train_probabilities),
                oof_auc=roc_auc_score(labels, oof_probabilities),
                oof_logloss=binary_logloss(labels, oof_probabilities),
                test_positive_count=sum(test_predictions),
                submission_path=str(submission_path),
            )
        )

    results.sort(
        key=lambda result: (
            -result.oof_accuracy,
            -result.oof_auc,
            result.oof_logloss,
            abs(result.threshold - 0.5),
        )
    )
    if not results:
        raise RuntimeError("no EML candidate remained valid on the train+test inference path")
    return results[0], results, skipped


def main() -> None:
    train_rows, labels, feature_names = load_unit_interval_rows(TRAIN_PATH)
    assert labels is not None
    test_rows, _, _ = load_unit_interval_rows(TEST_PATH)
    folds = stratified_folds(labels, FOLDS, SEED)

    catboost_result, catboost_candidates = evaluate_catboost_family(
        train_rows, labels, test_rows, feature_names, folds
    )
    eml_result, eml_candidates, skipped_eml_candidates = evaluate_eml_family(
        train_rows, labels, test_rows, folds
    )

    shutil.copyfile(catboost_result.submission_path, SUBMISSIONS_DATA_DIR / "catboost_best_submission.csv")
    shutil.copyfile(eml_result.submission_path, SUBMISSIONS_DATA_DIR / "eml_best_submission.csv")

    payload = {
        "experiment": "prepare_kaggle_submissions",
        "competition": "titanic",
        "metric_target": "accuracy",
        "train_dataset": str(TRAIN_PATH),
        "test_dataset": str(TEST_PATH),
        "folds": FOLDS,
        "seed": SEED,
        "selected_submissions": {
            "catboost": asdict(catboost_result),
            "eml": asdict(eml_result),
        },
        "best_aliases": {
            "catboost": str(SUBMISSIONS_DATA_DIR / "catboost_best_submission.csv"),
            "eml": str(SUBMISSIONS_DATA_DIR / "eml_best_submission.csv"),
        },
        "family_candidates": {
            "catboost": [asdict(row) for row in catboost_candidates],
            "eml": [asdict(row) for row in eml_candidates],
        },
        "skipped_eml_candidates": skipped_eml_candidates,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"saved {SUMMARY_PATH}")
    for result in (catboost_result, eml_result):
        print(
            result.family,
            result.candidate_id,
            f"oof_accuracy={result.oof_accuracy:.6f}",
            f"oof_auc={result.oof_auc:.6f}",
            f"oof_logloss={result.oof_logloss:.6f}",
            f"threshold={result.threshold:.6f}",
            f"submission={result.submission_path}",
        )
    for skipped in skipped_eml_candidates:
        print(
            "skipped_eml",
            skipped["candidate_id"],
            skipped["reason"],
        )


if __name__ == "__main__":
    main()
