"""Screening utilities for random full-binary EML trees."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


LeafLabel = str


@dataclass(frozen=True)
class RandomEMLTreeScreeningResult:
    """Summary for a random-tree survival screening run."""

    seed: int
    n_trees: int
    height: int
    survived: int
    failed_at_x0: int
    failed_at_x1: int


def generate_random_leaf_assignment(height: int, rng: random.Random) -> list[LeafLabel]:
    """Generate leaves for a full binary tree of the requested height."""

    if height < 0:
        raise ValueError("height must be non-negative")

    return [rng.choice(("x", "1")) for _ in range(2**height)]


def evaluate_full_binary_eml_tree(leaves: list[LeafLabel], x_value: float) -> float:
    """Evaluate a full binary tree whose internal nodes are eml(left, right)."""

    if not leaves:
        raise ValueError("tree must contain at least one leaf")
    if len(leaves) & (len(leaves) - 1):
        raise ValueError("number of leaves must be a power of two")

    level = [x_value if leaf == "x" else 1.0 for leaf in leaves]
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1]
            next_level.append(math.exp(left) - math.log(right))
        level = next_level
    return level[0]


def x_equals_one_value_by_height(height: int) -> float:
    """Return the deterministic value for any full tree of a given height at x=1."""

    if height < 0:
        raise ValueError("height must be non-negative")

    value = 1.0
    for _ in range(height):
        value = math.exp(value) - math.log(value)
    return value


def first_x_equals_one_overflow_height(max_height: int = 32) -> int | None:
    """Return the first height at which the x=1 recurrence overflows."""

    if max_height < 0:
        raise ValueError("max_height must be non-negative")

    for height in range(max_height + 1):
        try:
            x_equals_one_value_by_height(height)
        except OverflowError:
            return height
    return None


def count_surviving_random_trees(
    n_trees: int,
    height: int,
    seed: int,
) -> RandomEMLTreeScreeningResult:
    """Count how many random trees survive evaluation at x=0 and x=1."""

    if n_trees < 0:
        raise ValueError("n_trees must be non-negative")

    rng = random.Random(seed)
    survived = 0
    failed_at_x0 = 0
    failed_at_x1 = 0

    for _ in range(n_trees):
        leaves = generate_random_leaf_assignment(height, rng)

        ok_x0 = True
        try:
            evaluate_full_binary_eml_tree(leaves, 0.0)
        except Exception:
            ok_x0 = False
            failed_at_x0 += 1

        ok_x1 = True
        try:
            evaluate_full_binary_eml_tree(leaves, 1.0)
        except Exception:
            ok_x1 = False
            failed_at_x1 += 1

        if ok_x0 and ok_x1:
            survived += 1

    return RandomEMLTreeScreeningResult(
        seed=seed,
        n_trees=n_trees,
        height=height,
        survived=survived,
        failed_at_x0=failed_at_x0,
        failed_at_x1=failed_at_x1,
    )
