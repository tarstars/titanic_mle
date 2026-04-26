import math

import pytest

from titanic_mle import bootstrap_mle_tree, mle


def test_mle_matches_definition() -> None:
    assert math.isclose(mle(0.0, 1.0), 1.0)
    assert math.isclose(mle(1.0, math.e), math.e - 1.0)


def test_mle_rejects_non_positive_y() -> None:
    with pytest.raises(ValueError):
        mle(0.0, 0.0)


def test_bootstrap_mle_tree_has_expected_branches() -> None:
    root = bootstrap_mle_tree()

    assert root.name == "titanic_mle_root"
    assert [child.name for child in root.children] == [
        "function_knowledge",
        "dataset_knowledge",
        "modeling_context",
    ]
