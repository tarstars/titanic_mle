"""Small binary-classification metrics used in local experiments."""

from __future__ import annotations

import math


def sigmoid(value: float) -> float:
    """Numerically stable sigmoid."""

    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)

    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def binary_logloss(y_true: list[int], probabilities: list[float], eps: float = 1e-15) -> float:
    """Binary log loss with probability clipping."""

    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length")
    if not y_true:
        raise ValueError("inputs must be non-empty")

    total = 0.0
    for label, probability in zip(y_true, probabilities):
        p = min(max(probability, eps), 1.0 - eps)
        total += -(label * math.log(p) + (1 - label) * math.log(1.0 - p))
    return total / len(y_true)


def roc_auc_score(y_true: list[int], scores: list[float]) -> float:
    """ROC AUC via pairwise ranking."""

    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    if not y_true:
        raise ValueError("inputs must be non-empty")

    positives = [score for label, score in zip(y_true, scores) if label == 1]
    negatives = [score for label, score in zip(y_true, scores) if label == 0]
    if not positives or not negatives:
        raise ValueError("ROC AUC requires both positive and negative examples")

    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5

    return wins / (len(positives) * len(negatives))
