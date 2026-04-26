import math
import random

import pytest

from titanic_mle.random_eml_trees import (
    count_surviving_random_trees,
    evaluate_full_binary_eml_tree,
    first_x_equals_one_overflow_height,
    generate_random_leaf_assignment,
    x_equals_one_value_by_height,
)


def test_x_equals_one_value_is_leaf_assignment_independent() -> None:
    rng = random.Random(123)
    leaves_a = generate_random_leaf_assignment(height=3, rng=rng)
    leaves_b = generate_random_leaf_assignment(height=3, rng=rng)

    assert math.isclose(
        evaluate_full_binary_eml_tree(leaves_a, 1.0),
        evaluate_full_binary_eml_tree(leaves_b, 1.0),
    )


def test_x_equals_one_overflow_starts_at_height_four() -> None:
    assert first_x_equals_one_overflow_height(10) == 4
    assert math.isfinite(x_equals_one_value_by_height(3))
    with pytest.raises(OverflowError):
        x_equals_one_value_by_height(4)


def test_height_ten_random_screening_has_no_survivors_for_fixed_seed() -> None:
    result = count_surviving_random_trees(n_trees=1000, height=10, seed=20260420)

    assert result.survived == 0
    assert result.failed_at_x0 == 1000
    assert result.failed_at_x1 == 1000
