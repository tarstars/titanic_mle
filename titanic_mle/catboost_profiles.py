"""Reusable CatBoost profiles for Titanic benchmarks."""

from __future__ import annotations


CATBOOST_BASELINE_PROFILE_ID = "baseline_depth6_lr005"
CATBOOST_BASELINE_PARAMS: dict[str, object] = {
    "iterations": 400,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.0,
}


CATBOOST_TUNED_BALANCED_PROFILE_ID = "d3_lr003_l280_b8"
CATBOOST_TUNED_BALANCED_PARAMS: dict[str, object] = {
    "iterations": 800,
    "depth": 3,
    "learning_rate": 0.03,
    "l2_leaf_reg": 80.0,
    "random_strength": 8.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 3.0,
}


CATBOOST_TUNED_BEST_AUC_PROFILE_ID = "d3_lr005_l230_b5"
CATBOOST_TUNED_BEST_AUC_PARAMS: dict[str, object] = {
    "iterations": 350,
    "depth": 3,
    "learning_rate": 0.05,
    "l2_leaf_reg": 30.0,
    "random_strength": 5.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
}


CATBOOST_TUNED_BEST_LOGLOSS_PROFILE_ID = "d4_lr006_l230_b3"
CATBOOST_TUNED_BEST_LOGLOSS_PARAMS: dict[str, object] = {
    "iterations": 220,
    "depth": 4,
    "learning_rate": 0.06,
    "l2_leaf_reg": 30.0,
    "random_strength": 3.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
}
