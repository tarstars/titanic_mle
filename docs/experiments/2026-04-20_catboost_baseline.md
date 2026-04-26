# CatBoost Baseline On Normalized Titanic

## Goal

Measure how a strong off-the-shelf tree ensemble compares with the current stacked EML frontier on the same normalized Titanic train set.

The comparison target was:

- same rows: `data/interim/titanic_unit_interval_train.csv`
- same binary target: `Survived`
- same main metric: `ROC AUC`

## Idea

Use `CatBoostClassifier` as a baseline in two feature regimes:

- `all_unit_features`: all normalized Titanic features except `PassengerId`
- `stacked_support_features`: only `pclass_unit`, `sex_unit`, `age_unit`, `fare_unit`

The second regime is important because the current best stacked EML expressions are ultimately built from those four base features.

## Command

```bash
uv run python scripts/benchmark_catboost_on_unit_interval.py
```

## Results

Current stacked EML frontier from [meta_stacked_exact_search_top5_height_le_3.json](../../data/processed/meta_stacked_exact_search_top5_height_le_3.json):

- best stacked `ROC AUC = 0.8533005251440684`
- best stacked-`AUC` expression:

```text
((ps_log (sf_prob sf_auc)) ((ps_log sf_auc) (sf_auc ps_prob)))
```

CatBoost configuration:

- `iterations = 400`
- `depth = 6`
- `learning_rate = 0.05`
- `seed = 20260420`
- `folds = 5`

### All normalized features

- train `ROC AUC = 0.9844187731015456`
- train `logloss = 0.19360191130323057`
- 5-fold OOF `ROC AUC = 0.8625118503605705`
- 5-fold OOF `logloss = 0.44038659733227636`

### Only the four stacked-support features

- features: `pclass_unit`, `sex_unit`, `age_unit`, `fare_unit`
- train `ROC AUC = 0.9766241651487553`
- train `logloss = 0.22446416112063666`
- 5-fold OOF `ROC AUC = 0.8563789558900287`
- 5-fold OOF `logloss = 0.43961381692483853`

## Interpretation

Three points matter:

- On the same train set, CatBoost is clearly above the current stacked EML frontier on `ROC AUC`.
- Even when restricted to the same four base features that feed the stacked EML library, CatBoost still exceeds the current stacked EML train `ROC AUC`.
- The 5-fold OOF CatBoost numbers are strong, but they are only contextual here. They are not directly apples-to-apples with the current stacked EML frontier, because the EML frontier was discovered on the full train set rather than in a nested cross-validation loop.

The most useful immediate conclusion is:

- the current stacked EML search is interesting, but CatBoost is still the stronger baseline on this dataset

## Output Artifacts

- `data/processed/catboost_unit_interval_benchmark.json`

## Status

This CatBoost baseline is reproducible and fixed to the current normalized Titanic train set.

## Next Step

The right next comparison is not another raw train-set number. It is a fairer evaluation protocol:

- holdout or cross-validated comparison for EML and CatBoost under the same split logic
- then measure whether deeper stacked search closes the gap
