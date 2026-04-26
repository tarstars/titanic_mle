# 2026-04-20 Discrete Counting

## Goal

Count valid EML trees on the discrete two-input domain:

- `x0 = sex ∈ {0, 1}`
- `x1 = embarked ∈ {0, 0.5, 1}`
- grammar: `T := 1 | x0 | x1 | (T T)`
- operator: `eml(a, b) = exp(a) - ln(b)`

A tree is considered valid if it evaluates without error on all 6 points of the discrete domain.

## Implementations

- Python reference:
  - [two_input_discrete_trees.py](/home/tarstars/prj/titanic_mle/titanic_mle/two_input_discrete_trees.py)
  - [count_sex_embarked_pair_trees.py](/home/tarstars/prj/titanic_mle/scripts/count_sex_embarked_pair_trees.py)
- Rust implementation:
  - [discrete_domain.rs](/home/tarstars/prj/titanic_mle/rust/eml_tree_search/src/discrete_domain.rs)
  - [count_sex_embarked.rs](/home/tarstars/prj/titanic_mle/rust/eml_tree_search/src/bin/count_sex_embarked.rs)
  - [branch_bound_sex_embarked_logloss.rs](/home/tarstars/prj/titanic_mle/rust/eml_tree_search/src/bin/branch_bound_sex_embarked_logloss.rs)

## Commands

```bash
cargo run --manifest-path rust/eml_tree_search/Cargo.toml --bin count-sex-embarked
cargo run --manifest-path rust/eml_tree_search/Cargo.toml --bin branch-bound-sex-embarked-logloss
```

## Results

Exact valid-tree counts by height:

- `0`: `3`
- `1`: `3`
- `2`: `21`
- `3`: `543`
- `4`: `144,123`
- `5`: `5,355,379,950`

Total valid trees with height `< 6`:

- `5,355,524,643`

Exact branch-and-bound best logloss on the grouped `(sex, embarked)` problem:

- best `logloss`: `0.5046670394856749`
- best `ROC AUC`: `0.7883206041819789`
- best expression:

```text
(((x1 ((x1 1) (1 1))) (((x1 1) (1 1)) (x0 1))) ((1 ((x0 1) 1)) (1 (1 1))))
```

For comparison, the exact best expression with height `<= 4` had:

- `logloss = 0.5161671231442113`
- `ROC AUC = 0.7883206041819789`

So height `5` improved logloss without improving AUC.

## Status

- counting: exact
- best-logloss search on grouped `(sex, embarked)`: exact via branch-and-bound

## Next Step

Move from the discrete `sex × embarked` toy domain to richer feature pairs on grouped Titanic rows.
