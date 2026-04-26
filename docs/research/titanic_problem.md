# Titanic Problem Notes

## Dataset Shape

The local source dataset provides:

- `train.csv`
- `test.csv`
- `gender_submission.csv`

## Core Prediction Task

The Titanic challenge predicts passenger survival.

Typical target:

- `Survived` in `train.csv`

Typical identifier:

- `PassengerId`

## Common Feature Families

The raw Titanic tables usually include:

- passenger class
- name
- sex
- age
- number of siblings or spouses aboard
- number of parents or children aboard
- ticket
- fare
- cabin
- embarked

## Practical Modeling Constraints

- several fields contain missing values
- categorical and numeric fields are mixed
- `test.csv` does not include the target
- the final output format must align with Kaggle submission expectations

## Project-Specific Framing

This project is not starting from a standard off-the-shelf classifier first.

Instead, it is organized around:

1. understanding the custom `mle(x, y)` function
2. deciding how that function should interact with Titanic features
3. building an `mle tree` that can represent the reasoning structure
