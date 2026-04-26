from titanic_mle.two_input_discrete_trees import (
    SEX_EMBARKED_DOMAIN,
    count_valid_two_input_trees_height_le_five,
)


def test_default_domain_has_all_six_pairs() -> None:
    assert SEX_EMBARKED_DOMAIN == (
        (0.0, 0.0),
        (0.0, 0.5),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 0.5),
        (1.0, 1.0),
    )


def test_two_input_valid_tree_counts_match_known_values() -> None:
    summary = count_valid_two_input_trees_height_le_five()

    assert summary.exact_height_counts == {
        0: 3,
        1: 3,
        2: 21,
        3: 543,
        4: 144123,
        5: 5355379950,
    }
    assert summary.unique_signature_counts == {
        0: 3,
        1: 3,
        2: 21,
        3: 538,
        4: 141708,
    }
    assert summary.left_safe_counts == {
        0: 3,
        1: 3,
        2: 21,
        3: 291,
        4: 77490,
    }
    assert summary.right_safe_counts == {
        0: 1,
        1: 3,
        2: 17,
        3: 434,
        4: 68375,
    }
    assert summary.total_count_le_5 == 5355524643
