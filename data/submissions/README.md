# Submissions

This directory stores Kaggle-ready Titanic submission CSVs generated from the current project models.

Expected format:

- `PassengerId`
- `Survived`

Current workflow:

- choose the best `CatBoost` candidate by OOF accuracy on normalized train data
- choose the best `EML` candidate by OOF accuracy on normalized train data
- tune a binary threshold on OOF probabilities
- retrain or re-evaluate on full train
- emit one submission CSV per selected family

Metadata about the chosen candidates and thresholds is saved in:

- `submission_summary.json`
