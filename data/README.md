# Data Directory

This repository uses a three-stage data layout:

- `data/raw/`: original copied source files
- `data/interim/`: cleaned or partially transformed working tables
- `data/processed/`: modeling-ready outputs
- `data/submissions/`: Kaggle-ready submission CSVs and submission metadata

For Titanic, the canonical raw files are:

- `data/raw/train.csv`
- `data/raw/test.csv`
- `data/raw/gender_submission.csv`

Raw files should stay as close to the source dataset as possible.
