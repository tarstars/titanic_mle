# CatBoost Tuning To Reduce Overfitting

## Goal

Replace the original aggressive CatBoost baseline with a shared-CV tuned profile that keeps the same Titanic feature matrix but materially shrinks the train/OOF gap.

## Idea

Run a small manual hyperparameter sweep over:

- shallower trees
- stronger `l2_leaf_reg`
- higher `random_strength`
- warmer Bayesian bagging / subsampling

Then keep two useful frontiers:

- the best `OOF ROC AUC`
- the best balance between `OOF` quality and low overfitting gap

## Commands

```bash
uv run python scripts/tune_catboost_shared_cv.py
```

Shared-CV comparison against the current top12 EML frontier:

```bash
uv run python scripts/benchmark_eml_catboost_cv.py \
  --output data/processed/eml_catboost_shared_cv_benchmark_top12_tuned_catboost.json \
  --eml-report exact_top5_auc=data/processed/meta_stacked_exact_search_top5_height_le_3.json \
  --eml-report meta_ga_auc_iter3=data/processed/ga_meta_stacked_top5__iter3__auc.json \
  --eml-report meta_ga_calibrated_logloss_iter4=data/processed/ga_meta_stacked_top5__iter4__calibrated_logloss.json \
  --eml-report meta_ga_top12_auc_iter1=data/processed/ga_meta_stacked_top12__lib12_iter1__auc_calibrated_logloss.json \
  --eml-report meta_ga_top12_calibrated_logloss_iter2=data/processed/ga_meta_stacked_top12__lib12_iter2__calibrated_logloss.json
```

## Results

Old CatBoost baseline on all normalized features:

- train `ROC AUC = 0.9840645937856176`
- OOF `ROC AUC = 0.8604267195006338`
- `AUC gap = 0.12363787428498385`
- train `logloss = 0.19293148773042124`
- OOF `logloss = 0.4478909087053915`

Best `OOF ROC AUC` tuned profile: `d3_lr005_l230_b5`

- `iterations = 350`
- `depth = 3`
- `learning_rate = 0.05`
- `l2_leaf_reg = 30.0`
- `random_strength = 5.0`
- `bootstrap_type = Bayesian`
- `bagging_temperature = 1.0`
- train `ROC AUC = 0.9105231201866232`
- OOF `ROC AUC = 0.8703783593774965`
- `AUC gap = 0.040144760809126656`
- train `logloss = 0.3578776848190013`
- OOF `logloss = 0.414767429466934`

Balanced tuned profile: `d3_lr003_l280_b8`

- `iterations = 800`
- `depth = 3`
- `learning_rate = 0.03`
- `l2_leaf_reg = 80.0`
- `random_strength = 8.0`
- `bootstrap_type = Bayesian`
- `bagging_temperature = 3.0`
- train `ROC AUC = 0.9033143727564205`
- OOF `ROC AUC = 0.8690521841945483`
- `AUC gap = 0.03426218856187213`
- train `logloss = 0.3708747960694452`
- OOF `logloss = 0.416813282665077`

Current best EML tree on the same shared-CV benchmark:

- OOF `ROC AUC = 0.8692519093727031`
- OOF `logloss = 0.4443458265611179`

## Interpretation

The tuning worked.

- CatBoost train/OOF `ROC AUC` gap dropped from `0.1236` to `0.0401` for the best-AUC tuned profile.
- That is roughly a `67.5%` reduction in the ranking overfit gap.
- OOF `ROC AUC` improved from `0.8604` to `0.8704`.
- OOF `logloss` improved from `0.4479` to `0.4148`.

The important consequence for the project is uncomfortable but useful:

- once CatBoost is regularized properly, it no longer looks obviously overfit
- the tuned CatBoost frontier now slightly exceeds the current best EML tree on shared-CV `ROC AUC`
- it also beats the current best EML tree clearly on shared-CV `logloss`

So the right comparison target is no longer the old untuned CatBoost. It is the tuned shallow CatBoost profile.

## Output Artifacts

- `data/processed/catboost_shared_cv_tuning.json`
- `data/processed/eml_catboost_shared_cv_benchmark_top12_tuned_catboost.json`

## Next Step

The next EML loop should optimize against the new target:

- beat tuned CatBoost on shared-CV `ROC AUC`
- beat tuned CatBoost on shared-CV `logloss`
- do it without losing the structural interpretability advantage of the symbolic tree
