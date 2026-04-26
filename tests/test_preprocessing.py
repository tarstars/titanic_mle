from __future__ import annotations

from titanic_mle.preprocessing import (
    EMBARKED_MAPPING,
    SEX_MAPPING,
    TitanicUnitIntervalPreprocessor,
    load_csv_rows,
)
from titanic_mle.paths import RAW_DATA_DIR


def _sample_row(
    passenger_id: str,
    survived: str,
    pclass: str,
    sex: str,
    age: str,
    sibsp: str,
    parch: str,
    fare: str,
    cabin: str,
    embarked: str,
) -> dict[str, str]:
    return {
        "PassengerId": passenger_id,
        "Survived": survived,
        "Pclass": pclass,
        "Name": "Example, Mr. Test",
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": "A/5 21171",
        "Fare": fare,
        "Cabin": cabin,
        "Embarked": embarked,
    }


def test_preprocessor_encodes_reference_categories() -> None:
    train_rows = [
        _sample_row("1", "0", "3", "male", "20", "0", "0", "10", "", "S"),
        _sample_row("2", "1", "1", "female", "40", "2", "1", "30", "C85", "C"),
        _sample_row("3", "1", "2", "female", "", "1", "2", "20", "", "Q"),
    ]

    preprocessor = TitanicUnitIntervalPreprocessor.fit(train_rows)
    transformed = preprocessor.transform_row(train_rows[1])

    assert transformed["sex_unit"] == SEX_MAPPING["female"]
    assert transformed["embarked_unit"] == EMBARKED_MAPPING["C"]
    assert transformed["pclass_unit"] == 1.0
    assert transformed["cabin_known"] == 1.0


def test_preprocessor_imputes_missing_values_and_sets_flags() -> None:
    train_rows = [
        _sample_row("1", "0", "3", "male", "20", "0", "0", "10", "", "S"),
        _sample_row("2", "1", "1", "female", "40", "2", "1", "30", "C85", "C"),
        _sample_row("3", "1", "2", "female", "", "1", "2", "", "", ""),
    ]

    preprocessor = TitanicUnitIntervalPreprocessor.fit(train_rows)
    transformed = preprocessor.transform_row(train_rows[2])

    assert transformed["age_missing"] == 1.0
    assert transformed["fare_missing"] == 1.0
    assert transformed["embarked_missing"] == 1.0
    assert 0.0 <= transformed["age_unit"] <= 1.0
    assert 0.0 <= transformed["fare_unit"] <= 1.0
    assert 0.0 <= transformed["embarked_unit"] <= 1.0


def test_real_titanic_rows_transform_to_unit_interval() -> None:
    train_rows = load_csv_rows(RAW_DATA_DIR / "train.csv")
    test_rows = load_csv_rows(RAW_DATA_DIR / "test.csv")
    preprocessor = TitanicUnitIntervalPreprocessor.fit(train_rows)

    for row in preprocessor.transform_rows(train_rows) + preprocessor.transform_rows(test_rows):
        for key, value in row.items():
            if key in {"PassengerId", "Survived"}:
                continue
            assert 0.0 <= float(value) <= 1.0, f"{key} escaped [0, 1]: {value}"
