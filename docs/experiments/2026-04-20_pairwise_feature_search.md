# 2026-04-20 Pairwise Feature Search

## Goal

Screen all pairwise combinations of the normalized Titanic features and save the top expressions for each pair.

The feature tables come from:

- [titanic_unit_interval_train.csv](/home/tarstars/prj/titanic_mle/data/interim/titanic_unit_interval_train.csv)
- [titanic_unit_interval_metadata.json](/home/tarstars/prj/titanic_mle/data/interim/titanic_unit_interval_metadata.json)

## Search Scope

- input space: all `13` normalized features from the unit-interval preprocessing
- number of feature pairs: `78`
- ranking metric: `logloss`
- companion metric: `ROC AUC`
- exact enumeration depth used in the final all-pairs report: `height <= 3`

## Why Height `<= 3`

An exact all-pairs run at `height <= 4` was attempted first, but for continuous-valued pairs it became too expensive. One process grew to roughly `20 GB RSS` and did not finish in a reasonable time.

So the saved all-pairs artifact is intentionally:

- exact for `height <= 3`
- complete for all `78` pairs

This is a mass-screening report, not the final deep search.

## Implementation

- [pairwise_top10_expressions.rs](/home/tarstars/prj/titanic_mle/rust/eml_tree_search/src/bin/pairwise_top10_expressions.rs)

## Command

```bash
cargo run --manifest-path rust/eml_tree_search/Cargo.toml --bin pairwise-top10-expressions /home/tarstars/prj/titanic_mle
```

## Output Artifacts

- [pairwise_top10_expressions_height_le_3_logloss.json](/home/tarstars/prj/titanic_mle/data/processed/pairwise_top10_expressions_height_le_3_logloss.json)
- [pairwise_top10_expressions_height_le_3_logloss.csv](/home/tarstars/prj/titanic_mle/data/processed/pairwise_top10_expressions_height_le_3_logloss.csv)

## Main Results

Best pair by top-1 logloss:

- pair: `sex_unit × age_unit`
- grouped points: `145`
- best expression:

```text
((x0 (1 x1)) ((1 1) (1 x1)))
```

- best `logloss`: `0.531675095952892`
- best `ROC AUC`: `0.7618370455586446`

Other strong pairs:

- `sex_unit × embarked_unit`
  - best `logloss`: `0.5358110883566476`
  - best `ROC AUC`: `0.7883206041819789`
- `sex_unit × family_size_unit`
  - best `logloss`: `0.5421997865300966`
  - best `ROC AUC`: `0.8128282150427678`

## Status

- all-pairs report: complete
- exactness: exact at `height <= 3`

## Next Step

Use the best pairwise expressions as seeds for a deeper optimizer rather than continuing flat exhaustive search across every pair.
