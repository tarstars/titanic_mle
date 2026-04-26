# Meta-GA Search Above Stacked Terminals

## Goal

Push the stacked EML frontier beyond the exact `height <= 3` search by running a genetic algorithm over the stacked grammar itself.

The grammar in this stage was:

- leaf = one of the 5 calibrated pairwise-model terminals
- internal node = `eml(left, right) = exp(left) - ln(right)`

## Situation Before This Iteration

The exact stacked search over the 5-terminal library had already established:

- best stacked `ROC AUC = 0.8533005251440684`
- best stacked calibrated `logloss = 0.4433304901456944`

That result was exact for `height <= 3`, but it was still a small grammar.

## Idea

Use the exact winners as seeds and let a GA search deeper trees:

- `height <= 5` and then `height <= 6`
- separate objectives for calibrated `logloss` and `ROC AUC`
- seed from:
  - exact top-10 by `logloss`
  - exact top-10 by `ROC AUC`
  - all 5 terminal ids
  - later, the best expressions from earlier meta-GA runs

## Commands

First probability-oriented run:

```bash
cargo run --manifest-path rust/eml_tree_search/Cargo.toml --bin ga-meta-stacked-expression -- \
  --repo-root /home/tarstars/prj/titanic_mle \
  --objective calibrated_logloss \
  --population 256 \
  --generations 180 \
  --max-height 5 \
  --seed 20260420 \
  --run-tag iter1
```

Ranking-oriented run with a strong AUC seed from the previous run:

```bash
cargo run --manifest-path rust/eml_tree_search/Cargo.toml --bin ga-meta-stacked-expression -- \
  --repo-root /home/tarstars/prj/titanic_mle \
  --objective auc_calibrated_logloss \
  --seed-expression '((ps_log ((sa_log (sf_auc ps_log)) sf_auc)) ((sf_prob (sa_log sf_prob)) (sf_prob (sf_auc sa_log))))' \
  --population 256 \
  --generations 180 \
  --max-height 5 \
  --seed 20260421 \
  --run-tag iter2
```

Pure AUC refinement at `height <= 6`:

```bash
cargo run --manifest-path rust/eml_tree_search/Cargo.toml --bin ga-meta-stacked-expression -- \
  --repo-root /home/tarstars/prj/titanic_mle \
  --objective auc \
  --seed-expression '((ps_log ((sa_log sf_auc) ((ps_log ps_prob) (sf_auc ps_prob)))) ((sf_prob (ps_log sf_prob)) (sf_prob (sf_auc sa_log))))' \
  --seed-expression '((ps_prob ((sa_log (sf_auc sf_prob)) (sf_prob (sa_log sa_log)))) ((sf_prob (sa_log sf_prob)) (sa_log (sf_auc sf_auc))))' \
  --population 320 \
  --generations 150 \
  --max-height 6 \
  --seed 20260422 \
  --run-tag iter3
```

Calibrated-logloss refinement at `height <= 6`:

```bash
cargo run --manifest-path rust/eml_tree_search/Cargo.toml --bin ga-meta-stacked-expression -- \
  --repo-root /home/tarstars/prj/titanic_mle \
  --objective calibrated_logloss \
  --seed-expression '((ps_prob ((sa_log (sf_auc sf_prob)) (sf_prob (sa_log sa_log)))) ((sf_prob (sa_log sf_prob)) (sa_log (sf_auc sf_auc))))' \
  --population 256 \
  --generations 120 \
  --max-height 6 \
  --seed 20260423 \
  --run-tag iter4
```

## Results

### Iteration 1: calibrated logloss, `height <= 5`

- best calibrated `logloss = 0.4343865931538471`
- best raw `logloss = 0.5754753530331754`
- best `ROC AUC = 0.8527998806974936`

Best expression:

```text
((ps_prob ((sa_log (sf_auc sf_prob)) (sf_prob (sa_log sa_log)))) ((sf_prob (sa_log sf_prob)) (sa_log (sf_auc sf_auc))))
```

This immediately improved the probability frontier from `0.4433304901456944` to `0.4343865931538471`.

### Iteration 2: AUC-oriented, `height <= 5`

- best `ROC AUC = 0.8578009991584913`
- best calibrated `logloss = 0.4334316014471639`
- best raw `logloss = 0.5330939582182510`

Best expression:

```text
((ps_log ((sa_log sf_auc) ((ps_log ps_prob) (sf_auc ps_prob)))) ((sf_prob (ps_log sf_prob)) (sf_prob (sf_auc sa_log))))
```

This beat the exact stacked AUC frontier by a large margin.

### Iteration 3: pure AUC, `height <= 6`

- best `ROC AUC = 0.8592922804887142`
- best calibrated `logloss = 0.4405770001100263`
- best raw `logloss = 0.5188647566897557`

Best expression:

```text
(((ps_log ((sa_log sf_auc) ((ps_log sf_auc) sa_log))) (sf_prob sf_auc)) ((sf_prob (ps_log sf_prob)) (sf_prob (sa_log sa_log))))
```

This became the new ranking frontier.

### Iteration 4: calibrated logloss, `height <= 6`

- best calibrated `logloss = 0.4311395588032405`
- best `ROC AUC = 0.8584880537713440`
- best raw `logloss = 0.5232803413598930`

Best expression:

```text
((ps_log ((sa_log (sf_prob sf_prob)) (sf_prob (sf_auc sf_auc)))) (((sf_prob (sf_prob sf_auc)) (sa_log (sf_auc sf_auc))) ((sa_log ps_prob) ((ps_prob sf_auc) ((sa_log ps_prob) (sf_prob ps_prob))))))
```

This became the new probability frontier, and it also retained a strong ranking score.

## Interpretation

Four conclusions matter:

- The stacked terminal library was strong enough to support a second search layer; the GA did not just rediscover the exact `height <= 3` winners.
- The probability frontier moved from `0.4433304901456944` to `0.4311395588032405`.
- The ranking frontier moved from `0.8533005251440684` to `0.8592922804887142`.
- The best probability model and the best ranking model are now both meta-GA expressions, not exact stacked-search expressions.

Context against the CatBoost baseline:

- CatBoost with the same four support features had 5-fold OOF `ROC AUC = 0.8563789558900287`
- the new best meta-GA train `ROC AUC` is `0.8592922804887142`

This is not a fair apples-to-apples win, because one number is OOF and the other is train. But it does show that the EML search has moved into a competitive range for that smaller feature regime.

## Output Artifacts

- `data/processed/ga_meta_stacked_top5__iter1__calibrated_logloss.json`
- `data/processed/ga_meta_stacked_top5__iter2__auc_calibrated_logloss.json`
- `data/processed/ga_meta_stacked_top5__iter3__auc.json`
- `data/processed/ga_meta_stacked_top5__iter4__calibrated_logloss.json`

## Status

These results are approximate because the search stage is genetic.

## Next Step

The next useful step is a fairer evaluation and a richer library:

- evaluate the stacked EML frontier under the same CV protocol as CatBoost
- expand the terminal library with additional pairwise winners or selected meta-GA winners
- then repeat the search with pruning or a larger seeded population
