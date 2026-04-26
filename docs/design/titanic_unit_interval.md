# Titanic Unit-Interval Preprocessing

This document fixes the first preprocessing scheme for the "large EML tree + log loss" experiment direction.

The goal is simple:

- every model feature should lie in `[0, 1]`
- the same train-fitted transformation should be applied to both `train.csv` and `test.csv`

## Included Features

The first feature set is:

- `pclass_unit`
- `sex_unit`
- `age_unit`
- `age_missing`
- `sibsp_unit`
- `parch_unit`
- `fare_unit`
- `fare_missing`
- `embarked_unit`
- `embarked_missing`
- `cabin_known`
- `family_size_unit`
- `is_alone`

## Exact Mapping Rules

### Passenger class

Higher passenger class gets the higher normalized value:

- `Pclass=1 -> 1.0`
- `Pclass=2 -> 0.5`
- `Pclass=3 -> 0.0`

### Sex

The first version uses:

- `male -> 0.0`
- `female -> 1.0`

### Embarked

`Embarked` is categorical and does not have a natural numeric geometry. The first version uses a stable ordinal coding in alphabetical order:

- `C -> 0.0`
- `Q -> 0.5`
- `S -> 1.0`

Missing `Embarked` values are imputed with the training-set mode, and `embarked_missing` records the fact of imputation.

### Age

The first version follows the project suggestion directly:

```text
age_unit = clip(age / 80.0, 0.0, 1.0)
```

Missing age values are imputed with the training-set median, and `age_missing` records the fact of imputation.

### SibSp and Parch

These are normalized by the training-set maxima:

```text
sibsp_unit = SibSp / train_max_sibsp
parch_unit = Parch / train_max_parch
```

### Fare

Fare is normalized by the training-set maximum:

```text
fare_unit = clip(fare / train_max_fare, 0.0, 1.0)
```

Missing fare values are imputed with the training-set median, and `fare_missing` records the fact of imputation.

### Cabin

The first version only uses presence:

- `cabin_known = 1.0` if `Cabin` is present
- `cabin_known = 0.0` otherwise

### Family features

Derived features:

```text
family_size_unit = (SibSp + Parch + 1) / train_max_family_size
is_alone = 1.0 if family_size == 1 else 0.0
```

## Deliberate Omissions In V1

These raw fields are not encoded yet:

- `Name`
- `Ticket`

They are postponed because the first goal is to get a stable unit-interval table for EML experiments without inventing fragile text encodings too early.

## Output Files

Running the preparation step writes:

- `data/interim/titanic_unit_interval_train.csv`
- `data/interim/titanic_unit_interval_test.csv`
- `data/interim/titanic_unit_interval_metadata.json`
