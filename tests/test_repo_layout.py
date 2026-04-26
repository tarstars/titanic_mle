from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_files_exist() -> None:
    expected = [
        ROOT / "pyproject.toml",
        ROOT / "README.md",
        ROOT / "data" / "README.md",
        ROOT / "data" / "interim" / "README.md",
        ROOT / "data" / "processed" / "README.md",
        ROOT / "data" / "submissions" / "README.md",
        ROOT / "docs" / "architecture.md",
        ROOT / "docs" / "codex_jupyter_workflow.md",
        ROOT / "docs" / "design" / "mle_tree.md",
        ROOT / "docs" / "design" / "titanic_unit_interval.md",
        ROOT / "docs" / "experiments" / "README.md",
        ROOT / "docs" / "experiments" / "2026-04-20_discrete_counting.md",
        ROOT / "docs" / "experiments" / "2026-04-20_pairwise_feature_search.md",
        ROOT / "docs" / "experiments" / "2026-04-20_ga_search.md",
        ROOT / "docs" / "experiments" / "2026-04-20_metric_optimization_iterations.md",
        ROOT / "docs" / "experiments" / "2026-04-20_meta_stacked_search.md",
        ROOT / "docs" / "experiments" / "2026-04-20_catboost_baseline.md",
        ROOT / "docs" / "experiments" / "2026-04-20_meta_ga_search.md",
        ROOT / "docs" / "experiments" / "2026-04-20_shared_cv_benchmark.md",
        ROOT / "docs" / "experiments" / "2026-04-20_top12_terminal_library.md",
        ROOT / "docs" / "experiments" / "2026-04-20_catboost_tuning.md",
        ROOT / "docs" / "experiments" / "2026-04-20_submission_preparation.md",
        ROOT / "docs" / "research" / "eml_titanic_feasibility_2026-04-20.md",
        ROOT / "docs" / "research" / "mle_function.md",
        ROOT / "docs" / "research" / "titanic_problem.md",
        ROOT / "models" / "mle_trees" / "README.md",
        ROOT / "scripts" / "install_kernel.sh",
        ROOT / "scripts" / "analyze_height2_sex_eml_trees.py",
        ROOT / "scripts" / "analyze_sex_embarked_eml_family.py",
        ROOT / "scripts" / "build_sex_eml_tree_catalog.py",
        ROOT / "scripts" / "benchmark_catboost_on_unit_interval.py",
        ROOT / "scripts" / "benchmark_eml_catboost_cv.py",
        ROOT / "scripts" / "count_sex_embarked_pair_trees.py",
        ROOT / "scripts" / "prepare_titanic_unit_interval.py",
        ROOT / "scripts" / "prepare_kaggle_submissions.py",
        ROOT / "scripts" / "select_pairwise_terminal_library_cv.py",
        ROOT / "scripts" / "start_jupyter_lab.sh",
        ROOT / "notebooks" / "titanic_starter.ipynb",
        ROOT / "rust" / "eml_tree_search" / "Cargo.toml",
        ROOT / "rust" / "eml_tree_search" / "README.md",
        ROOT / "rust" / "eml_tree_search" / "src" / "lib.rs",
        ROOT / "rust" / "eml_tree_search" / "src" / "discrete_domain.rs",
        ROOT / "rust" / "eml_tree_search" / "src" / "bin" / "count_sex_embarked.rs",
        ROOT / "rust" / "eml_tree_search" / "src" / "bin" / "branch_bound_sex_embarked_logloss.rs",
        ROOT / "rust" / "eml_tree_search" / "src" / "bin" / "ga_best_expression.rs",
        ROOT / "rust" / "eml_tree_search" / "src" / "bin" / "ga_meta_stacked_expression.rs",
        ROOT / "rust" / "eml_tree_search" / "src" / "bin" / "meta_stacked_exact_search.rs",
        ROOT / "rust" / "eml_tree_search" / "src" / "bin" / "pairwise_top10_expressions.rs",
        ROOT / "rust" / "eml_tree_search" / "src" / "bin" / "search_sex_embarked_metrics.rs",
        ROOT / "titanic_mle" / "__init__.py",
        ROOT / "titanic_mle" / "mle.py",
        ROOT / "titanic_mle" / "metrics.py",
        ROOT / "titanic_mle" / "paths.py",
        ROOT / "titanic_mle" / "preprocessing.py",
        ROOT / "titanic_mle" / "sexpr_trees.py",
        ROOT / "titanic_mle" / "submissions.py",
        ROOT / "titanic_mle" / "tree.py",
        ROOT / "titanic_mle" / "tree_catalog.py",
        ROOT / "titanic_mle" / "two_input_discrete_trees.py",
    ]
    for path in expected:
        assert path.exists(), f"missing expected file: {path}"


def test_pyproject_contains_stable_pins() -> None:
    content = (ROOT / "pyproject.toml").read_text()
    assert 'name = "titanic-mle"' in content
    assert "jupyter-mcp-server" in content
    assert "jupyter-collaboration" in content
    assert "pycrdt<0.12.50" in content
