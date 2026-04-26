# Top12 Shared-CV Terminal Library And Meta-GA

## Goal

Move beyond the original 5-terminal stacked grammar by:

- screening pairwise candidates under the same shared-CV protocol
- building a larger but still curated terminal library
- rerunning the stacked GA with multithreaded evaluation
- checking whether the new frontier actually improves OOF `ROC AUC` and `logloss`

## Situation Before This Iteration

The previous best shared-CV EML frontiers were:

- best OOF `ROC AUC = 0.8574761128686926`
- best OOF `logloss = 0.4325811024681582`

Those models came from the `top5` terminal library. The main question was whether a larger terminal set would add real signal, or just more train overfit.

## Idea

The old terminal library was too small and too train-driven. The new idea was:

1. evaluate all saved pairwise top-10 expressions under shared CV
2. keep the current 5 terminals as anchors
3. add 7 new terminals chosen from pairwise shared-CV results
4. search again over the stacked grammar with `EML_SEARCH_THREADS=20`

This is not “add more random leaves”. It is a library expansion driven by out-of-fold behavior.

## Commands

Shared-CV screening of pairwise candidates:

```bash
uv run python scripts/select_pairwise_terminal_library_cv.py
```

Balanced ranking/probability search on the new library:

```bash
EML_SEARCH_THREADS=20 cargo run --manifest-path rust/eml_tree_search/Cargo.toml --bin ga-meta-stacked-expression -- \
  --repo-root /home/tarstars/prj/titanic_mle \
  --terminal-library /home/tarstars/prj/titanic_mle/data/processed/meta_terminal_library_top12_shared_cv.json \
  --objective auc_calibrated_logloss \
  --seed-expression '(((ps_log ((sa_log sf_auc) ((ps_log sf_auc) sa_log))) (sf_prob sf_auc)) ((sf_prob (ps_log sf_prob)) (sf_prob (sa_log sa_log))))' \
  --seed-expression '((ps_log ((sa_log (sf_prob sf_prob)) (sf_prob (sf_auc sf_auc)))) (((sf_prob (sf_prob sf_auc)) (sa_log (sf_auc sf_auc))) ((sa_log ps_prob) ((ps_prob sf_auc) ((sa_log ps_prob) (sf_prob ps_prob))))))' \
  --population 320 \
  --generations 140 \
  --max-height 6 \
  --seed 20260424 \
  --run-tag lib12_iter1
```

Probability-oriented search on the same library:

```bash
EML_SEARCH_THREADS=20 cargo run --manifest-path rust/eml_tree_search/Cargo.toml --bin ga-meta-stacked-expression -- \
  --repo-root /home/tarstars/prj/titanic_mle \
  --terminal-library /home/tarstars/prj/titanic_mle/data/processed/meta_terminal_library_top12_shared_cv.json \
  --objective calibrated_logloss \
  --seed-expression '((ps_log ((sa_log (sf_prob sf_prob)) (sf_prob (sf_auc sf_auc)))) (((sf_prob (sf_prob sf_auc)) (sa_log (sf_auc sf_auc))) ((sa_log ps_prob) ((ps_prob sf_auc) ((sa_log ps_prob) (sf_prob ps_prob))))))' \
  --seed-expression '(((ps_log ((sa_log sf_auc) ((ps_prob sf_prob) sx_fs_sx))) (sf_prob sa_log)) ((sf_prob ((ps_log (ps_log pc_ag_mx)) (ps_prob (pc_ag_mx pc_fr_mx)))) (sf_prob ((sf_prob (sa_log sx_fs_sx)) sx_fs_sx))))' \
  --population 320 \
  --generations 140 \
  --max-height 6 \
  --seed 20260425 \
  --run-tag lib12_iter2
```

Shared-CV benchmark against the older EML frontiers and CatBoost:

```bash
uv run python scripts/benchmark_eml_catboost_cv.py \
  --output /home/tarstars/prj/titanic_mle/data/processed/eml_catboost_shared_cv_benchmark_top12.json \
  --eml-report exact_top5_auc=data/processed/meta_stacked_exact_search_top5_height_le_3.json \
  --eml-report meta_ga_auc_iter3=data/processed/ga_meta_stacked_top5__iter3__auc.json \
  --eml-report meta_ga_calibrated_logloss_iter4=data/processed/ga_meta_stacked_top5__iter4__calibrated_logloss.json \
  --eml-report meta_ga_top12_auc_iter1=data/processed/ga_meta_stacked_top12__lib12_iter1__auc_calibrated_logloss.json \
  --eml-report meta_ga_top12_calibrated_logloss_iter2=data/processed/ga_meta_stacked_top12__lib12_iter2__calibrated_logloss.json
```

## Terminal Library

The expanded `top12` library kept the old anchors:

- `ps_prob`
- `ps_log`
- `sf_auc`
- `sa_log`
- `sf_prob`

and added:

- `sx_fs_sx`
- `sx_sb_sx`
- `pc_ag_mx`
- `pc_fs_mx`
- `pc_ia_mx`
- `pc_fr_mx`
- `pc_em_mx`

The selection was shared-CV-driven, not train-score-driven.

## Results

### New train frontiers

Balanced `auc_calibrated_logloss` run:

- train `ROC AUC = 0.8719175747504767`
- train calibrated `logloss = 0.4434081654540695`

Best expression:

```text
(((ps_log ((sa_log sf_auc) ((ps_prob sf_prob) sx_fs_sx))) (sf_prob sa_log)) ((sf_prob ((ps_log (ps_log pc_ag_mx)) (ps_prob (pc_ag_mx pc_fr_mx)))) (sf_prob ((sf_prob (sa_log sx_fs_sx)) sx_fs_sx))))
```

Probability-oriented run:

- train `ROC AUC = 0.8618301217524686`
- train calibrated `logloss = 0.4253525510220375`

Best expression:

```text
((ps_log ((sa_log (sf_prob sa_log)) (sf_prob (sx_sb_sx sx_fs_sx)))) (((sf_prob (sf_prob sf_prob)) (sa_log (sf_auc sf_auc))) (((pc_ag_mx (sf_auc pc_ag_mx)) ps_prob) ((sx_sb_sx pc_em_mx) sf_prob))))
```

### Shared-CV benchmark

Old best shared-CV EML ranking model:

- OOF `ROC AUC = 0.8574761128686926`
- OOF `logloss = 0.4417421556787212`

Old best shared-CV EML probability model:

- OOF `ROC AUC = 0.8544030081274833`
- OOF `logloss = 0.4325811024681582`

New `top12` ranking frontier:

- OOF `ROC AUC = 0.8692519093727031`
- OOF `logloss = 0.4443458265611179`

New `top12` probability frontier:

- OOF `ROC AUC = 0.8586371819043662`
- OOF `logloss = 0.4266281421497300`

CatBoost on all normalized features, same shared-CV protocol:

- OOF `ROC AUC = 0.8625118503605705`
- OOF `logloss = 0.4403865973322764`

## Interpretation

Three conclusions matter:

- Expanding the terminal library worked. The new ranking model beat the previous EML OOF AUC frontier by about `0.0118`.
- The new probability model also improved OOF logloss, from `0.43258` to `0.42663`.
- Under the shared-CV benchmark used here, the new `top12` EML frontiers beat the CatBoost baseline on both OOF `ROC AUC` and OOF `logloss`.

This is the first iteration where the EML search is not just competitive in a narrow regime. It is ahead on the benchmark we are actually using.

## Output Artifacts

- `data/processed/pairwise_candidates_shared_cv.json`
- `data/processed/meta_terminal_library_top12_shared_cv.json`
- `data/processed/ga_meta_stacked_top12__lib12_iter1__auc_calibrated_logloss.json`
- `data/processed/ga_meta_stacked_top12__lib12_iter2__calibrated_logloss.json`
- `data/processed/eml_catboost_shared_cv_benchmark_top12.json`

## Status

The pairwise shared-CV screening is exact for the saved candidate pool. The stacked search stage is approximate because it is genetic.

## Next Step

The next useful iteration is no longer “grow deeper at any cost”. It is:

- test a second, more feature-diverse terminal library variant
- keep the best `top12` model as a seed
- rerun shared-CV benchmark immediately after search
