import math

from titanic_mle.metrics import binary_logloss, roc_auc_score, sigmoid


def test_sigmoid_zero_is_half() -> None:
    assert math.isclose(sigmoid(0.0), 0.5)


def test_binary_logloss_prefers_better_probabilities() -> None:
    y_true = [0, 1]
    good = [0.1, 0.9]
    bad = [0.4, 0.6]

    assert binary_logloss(y_true, good) < binary_logloss(y_true, bad)


def test_roc_auc_score_detects_perfect_ranking() -> None:
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]

    assert math.isclose(roc_auc_score(y_true, scores), 1.0)
