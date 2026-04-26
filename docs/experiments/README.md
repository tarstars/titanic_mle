# Experiment Track

This folder keeps short, dated markdown records of the main Titanic MLE experiments.

Current entries:

- [2026-04-20_discrete_counting.md](./2026-04-20_discrete_counting.md): exact counting and exact branch-and-bound search on the discrete `sex × embarked` domain
- [2026-04-20_pairwise_feature_search.md](./2026-04-20_pairwise_feature_search.md): all-pairs feature screening on normalized Titanic data
- [2026-04-20_ga_search.md](./2026-04-20_ga_search.md): seeded genetic algorithm run from pairwise candidates
- [2026-04-20_metric_optimization_iterations.md](./2026-04-20_metric_optimization_iterations.md): objective-specific GA iterations plus monotone calibration on the best saved expressions
- [2026-04-20_meta_stacked_search.md](./2026-04-20_meta_stacked_search.md): exact search over calibrated pairwise-model terminals plus multithreaded speedup
- [2026-04-20_catboost_baseline.md](./2026-04-20_catboost_baseline.md): CatBoost ROC AUC baseline on the same normalized Titanic train set
- [2026-04-20_meta_ga_search.md](./2026-04-20_meta_ga_search.md): deeper genetic search over the stacked-terminal grammar
- [2026-04-20_shared_cv_benchmark.md](./2026-04-20_shared_cv_benchmark.md): shared-fold benchmark for current EML frontiers and CatBoost
- [2026-04-20_top12_terminal_library.md](./2026-04-20_top12_terminal_library.md): shared-CV terminal selection, multithreaded top12 meta-GA, and the updated EML vs CatBoost benchmark
- [2026-04-20_catboost_tuning.md](./2026-04-20_catboost_tuning.md): CatBoost hyperparameter tuning focused on reducing overfitting without giving up shared-CV quality
- [2026-04-20_submission_preparation.md](./2026-04-20_submission_preparation.md): selection of Kaggle-ready CatBoost and EML submission CSVs using OOF accuracy and threshold tuning

The goal is not to duplicate code. Each note captures:

- hypothesis / task
- command used
- exact or approximate status
- key numeric result
- output artifacts
- next step
