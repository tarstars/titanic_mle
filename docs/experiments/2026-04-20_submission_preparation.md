# Kaggle Submission Preparation

## Goal

Prepare real Kaggle-ready Titanic submission files for both model families:

- tuned `CatBoost`
- stacked `EML`

Unlike the earlier model-selection notes, this step targets the competition metric directly:

- binary `Survived`
- judged by classification accuracy on hidden test labels

## Idea

For each family:

1. evaluate several candidate models on train via shared-CV
2. choose a probability threshold by maximizing OOF accuracy
3. keep the best family member by OOF accuracy
4. fit or evaluate on full train
5. emit `PassengerId,Survived` submission CSVs for `test.csv`

## Command

```bash
uv run python scripts/prepare_kaggle_submissions.py
```

## Results

Selected `CatBoost` submission:

- candidate: `d4_lr006_l230_b3`
- OOF `accuracy = 0.8327721661054994`
- OOF `ROC AUC = 0.8694889165841136`
- OOF `logloss = 0.4139430754448254`
- chosen threshold: `0.5451145089065665`
- predicted positives on test: `127`

Selected `EML` submission:

- candidate: `meta_ga_top12_auc_iter1`
- OOF `accuracy = 0.8237934904601572`
- OOF `ROC AUC = 0.8692519093727031`
- OOF `logloss = 0.4443458265611179`
- chosen threshold: `0.3001438137765018`
- predicted positives on test: `152`

One important operational detail showed up during submission generation:

- `meta_ga_top12_calibrated_logloss_iter2` looked good on train metrics, but failed on real test inference with a `log domain` error
- so it was excluded from the EML submission family

## Output Artifacts

Traced submission files:

- `data/submissions/catboost_d4_lr006_l230_b3_submission.csv`
- `data/submissions/eml_meta_ga_top12_auc_iter1_submission.csv`

Convenience aliases:

- `data/submissions/catboost_best_submission.csv`
- `data/submissions/eml_best_submission.csv`

Full selection trace:

- `data/submissions/submission_summary.json`

## Interpretation

For Kaggle upload today, the stronger file is the CatBoost submission.

It wins over the current EML family on the competition-relevant train proxy:

- higher OOF accuracy
- slightly higher OOF `ROC AUC`
- substantially lower OOF `logloss`

The EML submission is still worth keeping as a symbolic baseline, but it is not the stronger Kaggle candidate right now.

## Next Step

The next practical improvement loop is:

- push EML OOF accuracy above `0.833`
- explicitly penalize test-invalid trees during search
- then regenerate both submission families
