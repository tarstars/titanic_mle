# Meta-Stacked Exact Search

## Goal

Test whether the strongest pairwise EML expressions can be treated as reusable building blocks rather than final models.

The concrete target was:

- take saved pairwise winners,
- calibrate each of them into a probability-valued terminal,
- run an exact small-depth EML search above those terminals,
- check whether this stacked layer can beat the current pairwise frontiers on `logloss` and `ROC AUC`.

## Idea

Instead of searching directly over raw Titanic features again, use a higher-level grammar:

- terminal = calibrated probability output of a saved pairwise expression
- internal node = `eml(left, right) = exp(left) - ln(right)`

This keeps the search exact and small enough for `height <= 3`, while allowing the model to combine:

- a probability-oriented `pclass_unit × sex_unit` model,
- a ranking-oriented `sex_unit × fare_unit` model,
- an age-bearing `sex_unit × age_unit` model,
- later, an additional probability-oriented `sex_unit × fare_unit` model.

## Commands

Exact stacked search, current 5-terminal library:

```bash
cargo run --manifest-path rust/eml_tree_search/Cargo.toml --bin meta-stacked-exact-search -- \
  --repo-root /home/tarstars/prj/titanic_mle \
  --max-height 3
```

Single-thread benchmark:

```bash
/usr/bin/time -f 'elapsed=%E cpu=%P' \
  env EML_SEARCH_THREADS=1 \
  rust/eml_tree_search/target/debug/meta-stacked-exact-search \
  --repo-root /home/tarstars/prj/titanic_mle \
  --max-height 3
```

20-thread benchmark:

```bash
/usr/bin/time -f 'elapsed=%E cpu=%P' \
  env EML_SEARCH_THREADS=20 \
  rust/eml_tree_search/target/debug/meta-stacked-exact-search \
  --repo-root /home/tarstars/prj/titanic_mle \
  --max-height 3
```

## Results

### Pilot: 4-terminal library

This exact run used the initial terminal library:

- `ps_prob`: `pclass_unit × sex_unit` seeded calibrated-logloss winner
- `ps_log`: `pclass_unit × sex_unit` raw logloss winner
- `sf_auc`: `sex_unit × fare_unit` AUC winner
- `sa_log`: `sex_unit × age_unit` logloss-oriented winner

Result:

- total expressions: `163,220`
- valid expressions: `148,676`
- best raw `logloss = 0.4478717350536450`
- corresponding `ROC AUC = 0.8415353806495596`
- corresponding calibrated `logloss = 0.4474733088366046`
- best `ROC AUC = 0.8486349449823709`
- corresponding raw `logloss = 0.4646810376431634`

Best by `logloss`:

```text
((ps_prob (sa_log ps_prob)) ((ps_log sa_log) (ps_prob sf_auc)))
```

Best by `ROC AUC`:

```text
((ps_log (sf_auc sa_log)) ((ps_prob sf_auc) (ps_log sa_log)))
```

This already beat the pairwise frontier on both metrics.

### Expanded: 5-terminal library

Then the library was widened by adding:

- `sf_prob`: `sex_unit × fare_unit` calibrated-logloss winner

Result:

- total expressions: `819,030`
- valid expressions: `737,580`
- best raw `logloss = 0.4450016918784160`
- corresponding `ROC AUC = 0.8469519274811194`
- corresponding calibrated `logloss = 0.4433304901456944`
- best `ROC AUC = 0.8533005251440684`
- corresponding raw `logloss = 0.4686754918838812`

Best by `logloss`:

```text
((ps_prob (sa_log sf_auc)) ((sf_prob sf_prob) (ps_prob sf_prob)))
```

Best by `ROC AUC`:

```text
((ps_log (sf_prob sf_auc)) ((ps_log sf_auc) (sf_auc ps_prob)))
```

Compared with the pairwise frontier before stacking:

- best pairwise calibrated `logloss`: `0.4494937812644100`
- best stacked calibrated `logloss`: `0.4433304901456944`
- best pairwise `ROC AUC`: `0.8345796184450196`
- best stacked `ROC AUC`: `0.8533005251440684`

So the stacked exact search improved both fronts materially.

## Multithreading

The first attempt parallelized only candidate evaluation and produced no observable wall-clock improvement. The actual bottleneck was broader: exact-level construction plus evaluation.

The binary was then rewritten to:

- build exact-height work in chunks,
- evaluate each chunk inside worker threads,
- keep the thread count configurable via `EML_SEARCH_THREADS`,
- preserve exactly the same search result.

Measured on the 5-terminal, `height <= 3` exact search:

- `EML_SEARCH_THREADS=1`
  - `elapsed = 5:37.12`
  - `cpu = 99%`
- `EML_SEARCH_THREADS=20`
  - `elapsed = 0:50.66`
  - `cpu = 1546%`

This is about a `6.65x` wall-clock speedup.

## Interpretation

Three conclusions matter:

- Pairwise winners are not endpoints; they are useful terminals for a second search layer.
- Even a very small exact stacked grammar can move both `logloss` and `ROC AUC` beyond the best pairwise models.
- The stacked search is now fast enough to be practical in the current environment, because the exact `819,030`-expression run dropped from minutes to under a minute.

## Output Artifacts

- `data/processed/meta_stacked_exact_search_height_le_3.json`
- `data/processed/meta_stacked_exact_search_top5_height_le_3.json`

## Status

These results are exact for the chosen terminal library and `height <= 3`.

## Next Step

The next useful step is to turn the best stacked expressions into seeds for a deeper search:

- genetic search above the stacked grammar,
- or an exact / beam search at larger height with pruning,
- and then compare the new frontier against the current `top5 height <= 3` exact winner.
