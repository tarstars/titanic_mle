# Shared CV Benchmark For EML And CatBoost

## Goal

Evaluate the current EML frontier and CatBoost under the same fold logic instead of comparing:

- EML train metrics
- against CatBoost 5-fold OOF metrics

## Protocol

Shared settings:

- dataset: `data/interim/titanic_unit_interval_train.csv`
- folds: stratified `5`-fold split
- seed: `20260420`

CatBoost:

- trained from scratch on each fold

Stacked/meta EML:

- structure fixed from the saved report
- terminal calibrations fixed from the saved terminal library
- top-layer sigmoid-affine calibration refit on each training fold

This is not fully nested CV for EML, because structure selection and terminal calibration were still chosen from full-train artifacts. But it is much fairer than comparing pure train metrics against CatBoost OOF metrics.

## Command

```bash
uv run python scripts/benchmark_eml_catboost_cv.py
```

## Results

### EML models

Exact stacked baseline:

- model: `exact_top5_auc`
- train `ROC AUC = 0.8533005251440684`
- OOF `ROC AUC = 0.8337940327442772`
- train `logloss = 0.4677481460435091`
- OOF `logloss = 0.4685332380528265`

Best meta-GA ranking model:

- model: `meta_ga_auc_iter3`
- train `ROC AUC = 0.8592922804887142`
- OOF `ROC AUC = 0.8574761128686926`
- train `logloss = 0.4405770001100263`
- OOF `logloss = 0.4417421556787212`

Best meta-GA probability model:

- model: `meta_ga_calibrated_logloss_iter4`
- train `ROC AUC = 0.8584880537713440`
- OOF `ROC AUC = 0.8544030081274833`
- train `logloss = 0.4311395588032405`
- OOF `logloss = 0.4325811024681582`

### CatBoost models

All normalized features:

- train `ROC AUC = 0.9844187731015456`
- OOF `ROC AUC = 0.8625118503605705`
- train `logloss = 0.1936019113032306`
- OOF `logloss = 0.4403865973322764`

Only stacked-support features `pclass_unit`, `sex_unit`, `age_unit`, `fare_unit`:

- train `ROC AUC = 0.9766241651487553`
- OOF `ROC AUC = 0.8563789558900287`
- train `logloss = 0.2244641611206367`
- OOF `logloss = 0.4396138169248385`

## Interpretation

Four conclusions matter:

- The exact stacked baseline was over-optimistic on train. Its OOF AUC dropped hard.
- The meta-GA frontier generalizes much better. The best ranking model kept most of its strength on OOF.
- The best meta-GA AUC model beats CatBoost restricted to the same four support features on OOF `ROC AUC`.
- The best meta-GA probability model beats both CatBoost baselines on OOF `logloss`.

So the situation is now:

- CatBoost on all normalized features is still the strongest baseline by OOF `ROC AUC`
- the best current EML ranking model is close
- the best current EML probability model is already competitive to strong on OOF `logloss`

## Output Artifact

- `data/processed/eml_catboost_shared_cv_benchmark.json`

## Status

This benchmark is a shared-fold evaluation, but not a fully nested CV search for EML.

## Next Step

The next useful improvement step is now clearer:

- expand the terminal library, because the current 5-terminal meta-GA models generalize well
- then rerun meta search and benchmark again under the same shared-CV protocol
