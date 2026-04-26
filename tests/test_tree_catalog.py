from titanic_mle.tree_catalog import load_titanic_sex_benchmark_stats, summarize_catalog_rows
from titanic_mle.tree_catalog import iter_catalog_rows
from titanic_mle.paths import RAW_DATA_DIR


def test_small_catalog_summary_matches_known_survival_counts() -> None:
    benchmark = load_titanic_sex_benchmark_stats(RAW_DATA_DIR / "train.csv")
    rows, summary = summarize_catalog_rows(iter_catalog_rows(max_height=2, benchmark=benchmark))

    assert len(rows) == 38
    assert summary["survived_trees"] == 14
    assert summary["counts_by_exact_height"]["0"] == {"total": 2, "survived": 2}
    assert summary["counts_by_exact_height"]["1"] == {"total": 4, "survived": 2}
    assert summary["counts_by_exact_height"]["2"] == {"total": 32, "survived": 10}
