# eml-tree-search

Small Rust crate for the heavy part of the `titanic_mle` search workflow.

Current scope:

- count valid EML trees on the discrete `sex x embarked` domain
- keep exact-height counts
- keep signature counts up to height `4`
- derive the exact height-`5` count without materializing all height-`5` trees

Current grammar:

```text
T := 1 | x0 | x1 | (T T)
```

where:

- `x0 = sex`
- `x1 = embarked`
- `(A B)` means one internal `eml(A, B) = exp(A) - ln(B)` node

Run:

```bash
cargo run --bin count-sex-embarked
```

The binary prints a JSON summary with the same counts currently used in the
Python research code.

For metric search over Titanic grouped by `(sex, embarked)`:

```bash
cargo run --bin search-sex-embarked-metrics -- 1000000
```

The optional numeric argument is the number of height-`5` samples.

For exact branch-and-bound search of the best height-`<=5` tree by logloss:

```bash
cargo run --bin branch-bound-sex-embarked-logloss
```

For a seeded genetic algorithm on the best current feature pair:

```bash
cargo run --bin ga-best-expression -- --repo-root /home/tarstars/prj/titanic_mle
```

You can also target a specific pair:

```bash
cargo run --bin ga-best-expression -- --repo-root /home/tarstars/prj/titanic_mle --feature-a sex_unit --feature-b age_unit
```
