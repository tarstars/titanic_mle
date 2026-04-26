# Titanic MLE

Local research project for solving the Titanic challenge with a custom `mle` approach, where:

```text
mle(x, y) = exp(x) - ln(y)
```

For this project, `mle` is the function above, not maximum likelihood estimation. That naming collision matters and should stay explicit in the docs and code.

## Current Goals

The first stage of the project is:

- collect what we know about the `mle` function
- gather the Titanic dataset locally
- design an `mle tree` that can organize the modeling context

This repository is now structured to support those steps directly.

## Project Structure

```text
.
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── README.md
├── docs/
│   ├── architecture.md
│   ├── codex_jupyter_workflow.md
│   ├── design/
│   │   ├── mle_tree.md
│   │   └── titanic_unit_interval.md
│   ├── experiments/
│   ├── research/
│   │   ├── mle_function.md
│   │   └── titanic_problem.md
│   └── troubleshooting.md
├── models/
│   └── mle_trees/
├── notebooks/
│   └── titanic_starter.ipynb
├── rust/
│   └── eml_tree_search/
├── scripts/
│   ├── install_kernel.sh
│   ├── prepare_titanic_unit_interval.py
│   └── start_jupyter_lab.sh
├── tests/
│   ├── test_mle.py
│   └── test_repo_layout.py
├── titanic_mle/
│   ├── __init__.py
│   ├── mle.py
│   ├── paths.py
│   └── tree.py
├── pyproject.toml
└── README.md
```

## Data Layout

Canonical dataset location inside this repository:

- `data/raw/train.csv`
- `data/raw/test.csv`
- `data/raw/gender_submission.csv`

Planned downstream locations:

- `data/interim/` for cleaned and feature-engineered working tables
- `data/processed/` for modeling-ready datasets
- `data/submissions/` for Kaggle-ready output files

## Research And Design Files

- [`docs/research/mle_function.md`](./docs/research/mle_function.md): current known properties of `mle(x, y)`
- [`docs/research/titanic_problem.md`](./docs/research/titanic_problem.md): problem framing and dataset facts
- [`docs/research/eml_titanic_feasibility_2026-04-20.md`](./docs/research/eml_titanic_feasibility_2026-04-20.md): sourced feasibility note on EML/MLE trees for Titanic
- [`docs/design/mle_tree.md`](./docs/design/mle_tree.md): initial structure for the future `mle tree`
- [`docs/design/titanic_unit_interval.md`](./docs/design/titanic_unit_interval.md): first Titanic-to-`[0, 1]` preprocessing scheme
- [`docs/experiments/README.md`](./docs/experiments/README.md): markdown track of completed experiments and saved artifacts

## Code Skeleton

- [`titanic_mle/mle.py`](./titanic_mle/mle.py): the core `mle` function
- [`titanic_mle/preprocessing.py`](./titanic_mle/preprocessing.py): train-fitted Titanic normalization into `[0, 1]`
- [`titanic_mle/tree.py`](./titanic_mle/tree.py): lightweight tree primitives for organizing the approach
- [`titanic_mle/paths.py`](./titanic_mle/paths.py): canonical project paths

## Rust Search Engine

The heavy combinatorial search is now starting to move into Rust.

- [`rust/eml_tree_search`](./rust/eml_tree_search/README.md): Rust crate for discrete EML tree counting and future search kernels

The first Rust target is the exact counting problem for:

```text
T := 1 | x0 | x1 | (T T)
```

with:

- `x0 = sex`
- `x1 = embarked`
- validity checked on the full discrete domain `sex x embarked = {0, 1} x {0, 0.5, 1}`

When a Rust toolchain is available locally, run:

```bash
cargo run --manifest-path rust/eml_tree_search/Cargo.toml --bin count-sex-embarked
```

For exact metric ranking on unique behaviors up to height `4`, plus sampled
height-`5` search:

```bash
cargo run --manifest-path rust/eml_tree_search/Cargo.toml --bin search-sex-embarked-metrics -- 1000000
```

For exact branch-and-bound search on grouped Titanic `(sex, embarked)` data:

```bash
cargo run --manifest-path rust/eml_tree_search/Cargo.toml --bin branch-bound-sex-embarked-logloss
```

For seeded GA search on the current best pairwise candidates:

```bash
cargo run --manifest-path rust/eml_tree_search/Cargo.toml --bin ga-best-expression -- --repo-root /home/tarstars/prj/titanic_mle
```

## Jupyter + MCP

This repo still supports direct notebook work through JupyterLab and MCP.

Default local endpoint:

- URL: `http://127.0.0.1:8894`
- token: `titanic-mle-local-token`

Start it with:

```bash
uv sync --dev
./scripts/install_kernel.sh
./scripts/start_jupyter_lab.sh
```

## Next Logical Steps

- inspect raw Titanic columns and missingness
- decide how `mle(x, y)` maps onto Titanic features and targets
- define the first executable version of the `mle tree`
- add baseline experiments in notebooks and code

## Prepare Unit-Interval Features

To generate the first normalized feature tables for EML experiments:

```bash
uv run python scripts/prepare_titanic_unit_interval.py
```

This writes:

- `data/interim/titanic_unit_interval_train.csv`
- `data/interim/titanic_unit_interval_test.csv`
- `data/interim/titanic_unit_interval_metadata.json`

## Prepare Kaggle Submissions

To generate the current best `CatBoost` and `EML` submission files:

```bash
uv run python scripts/prepare_kaggle_submissions.py
```

This writes:

- `data/submissions/catboost_best_submission.csv`
- `data/submissions/eml_best_submission.csv`
- `data/submissions/submission_summary.json`
