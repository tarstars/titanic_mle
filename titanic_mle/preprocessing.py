"""Preprocessing helpers for Titanic EML/MLE experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

from .paths import INTERIM_DATA_DIR, RAW_DATA_DIR


SEX_MAPPING = {"male": 0.0, "female": 1.0}

# This is an arbitrary but stable ordinal coding for a categorical variable.
# The first version uses alphabetical order to avoid pretending there is a
# domain-derived numeric spacing between ports.
EMBARKED_MAPPING = {"C": 0.0, "Q": 0.5, "S": 1.0}

# Higher passenger class gets the higher normalized value.
PCLASS_MAPPING = {1: 1.0, 2: 0.5, 3: 0.0}


@dataclass
class TitanicUnitIntervalStats:
    age_fill: float
    fare_fill: float
    embarked_fill: str
    max_sibsp: int
    max_parch: int
    max_fare: float
    max_family_size: int


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load a CSV file into a list of row dictionaries."""

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    """Write row dictionaries to a CSV file."""

    if not rows:
        raise ValueError("cannot write an empty row set")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_optional_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class TitanicUnitIntervalPreprocessor:
    """Train-fitted unit-interval preprocessing for Titanic."""

    stats: TitanicUnitIntervalStats

    @classmethod
    def fit(cls, train_rows: list[dict[str, str]]) -> "TitanicUnitIntervalPreprocessor":
        ages = [_parse_optional_float(row["Age"]) for row in train_rows]
        fares = [_parse_optional_float(row["Fare"]) for row in train_rows]
        present_ages = [value for value in ages if value is not None]
        present_fares = [value for value in fares if value is not None]
        embarked_values = [row["Embarked"].strip() for row in train_rows if row["Embarked"].strip()]
        family_sizes = [1 + int(row["SibSp"]) + int(row["Parch"]) for row in train_rows]

        if not present_ages or not present_fares or not embarked_values:
            raise ValueError("training rows are missing required statistics for preprocessing")

        embarked_fill = max(
            sorted(set(embarked_values)),
            key=lambda port: embarked_values.count(port),
        )

        stats = TitanicUnitIntervalStats(
            age_fill=float(median(present_ages)),
            fare_fill=float(median(present_fares)),
            embarked_fill=embarked_fill,
            max_sibsp=max(int(row["SibSp"]) for row in train_rows),
            max_parch=max(int(row["Parch"]) for row in train_rows),
            max_fare=max(present_fares),
            max_family_size=max(family_sizes),
        )
        return cls(stats=stats)

    def transform_row(self, row: dict[str, str]) -> dict[str, float | str]:
        """Transform one Titanic row into unit-interval features."""

        age_value = _parse_optional_float(row["Age"])
        fare_value = _parse_optional_float(row["Fare"])
        embarked_value = row["Embarked"].strip() or self.stats.embarked_fill

        sibsp = int(row["SibSp"])
        parch = int(row["Parch"])
        family_size = 1 + sibsp + parch

        transformed: dict[str, float | str] = {
            "PassengerId": row["PassengerId"],
            "pclass_unit": PCLASS_MAPPING[int(row["Pclass"])],
            "sex_unit": SEX_MAPPING[row["Sex"].strip().lower()],
            "age_unit": _clip01((age_value if age_value is not None else self.stats.age_fill) / 80.0),
            "age_missing": 1.0 if age_value is None else 0.0,
            "sibsp_unit": _clip01(sibsp / self.stats.max_sibsp) if self.stats.max_sibsp else 0.0,
            "parch_unit": _clip01(parch / self.stats.max_parch) if self.stats.max_parch else 0.0,
            "fare_unit": _clip01(
                (fare_value if fare_value is not None else self.stats.fare_fill) / self.stats.max_fare
            )
            if self.stats.max_fare
            else 0.0,
            "fare_missing": 1.0 if fare_value is None else 0.0,
            "embarked_unit": EMBARKED_MAPPING[embarked_value],
            "embarked_missing": 1.0 if not row["Embarked"].strip() else 0.0,
            "cabin_known": 1.0 if row["Cabin"].strip() else 0.0,
            "family_size_unit": _clip01(family_size / self.stats.max_family_size)
            if self.stats.max_family_size
            else 0.0,
            "is_alone": 1.0 if family_size == 1 else 0.0,
        }

        if "Survived" in row:
            transformed["Survived"] = row["Survived"]

        return transformed

    def transform_rows(self, rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
        """Transform a full row set."""

        return [self.transform_row(row) for row in rows]

    def metadata(self) -> dict[str, object]:
        """Return preprocessing metadata for experiment traceability."""

        return {
            "feature_set": [
                "pclass_unit",
                "sex_unit",
                "age_unit",
                "age_missing",
                "sibsp_unit",
                "parch_unit",
                "fare_unit",
                "fare_missing",
                "embarked_unit",
                "embarked_missing",
                "cabin_known",
                "family_size_unit",
                "is_alone",
            ],
            "mappings": {
                "sex_unit": SEX_MAPPING,
                "embarked_unit": EMBARKED_MAPPING,
                "pclass_unit": {"1": 1.0, "2": 0.5, "3": 0.0},
            },
            "scaling": {
                "age_unit": "clip(impute_median(age) / 80.0, 0.0, 1.0)",
                "sibsp_unit": "sibsp / train_max_sibsp",
                "parch_unit": "parch / train_max_parch",
                "fare_unit": "clip(impute_median(fare) / train_max_fare, 0.0, 1.0)",
                "family_size_unit": "(sibsp + parch + 1) / train_max_family_size",
            },
            "missing_indicators": ["age_missing", "fare_missing", "embarked_missing"],
            "train_stats": asdict(self.stats),
        }


def prepare_titanic_unit_interval(
    raw_train_path: Path | None = None,
    raw_test_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Prepare unit-interval train/test CSVs for Titanic experiments."""

    train_path = raw_train_path or RAW_DATA_DIR / "train.csv"
    test_path = raw_test_path or RAW_DATA_DIR / "test.csv"
    target_dir = output_dir or INTERIM_DATA_DIR

    train_rows = load_csv_rows(train_path)
    test_rows = load_csv_rows(test_path)

    preprocessor = TitanicUnitIntervalPreprocessor.fit(train_rows)
    transformed_train = preprocessor.transform_rows(train_rows)
    transformed_test = preprocessor.transform_rows(test_rows)

    target_dir.mkdir(parents=True, exist_ok=True)
    output_train_path = target_dir / "titanic_unit_interval_train.csv"
    output_test_path = target_dir / "titanic_unit_interval_test.csv"
    metadata_path = target_dir / "titanic_unit_interval_metadata.json"

    write_csv_rows(output_train_path, transformed_train)
    write_csv_rows(output_test_path, transformed_test)
    metadata_path.write_text(json.dumps(preprocessor.metadata(), indent=2) + "\n", encoding="utf-8")

    return {
        "train": output_train_path,
        "test": output_test_path,
        "metadata": metadata_path,
    }
