"""Enumeration and serialization utilities for EML s-expression trees."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator


TreeExpr = str | tuple["TreeExpr", "TreeExpr"]


@dataclass(frozen=True)
class TreeShapeSummary:
    """Shape summary for one tree."""

    height: int
    leaf_count: int
    internal_node_count: int
    total_node_count: int


def tree_height(tree: TreeExpr) -> int:
    """Return tree height with leaves at height 0."""

    if tree in {"x", "1"}:
        return 0

    left, right = tree
    return 1 + max(tree_height(left), tree_height(right))


def tree_leaf_count(tree: TreeExpr) -> int:
    """Return the number of leaves."""

    if tree in {"x", "1"}:
        return 1

    left, right = tree
    return tree_leaf_count(left) + tree_leaf_count(right)


def tree_internal_node_count(tree: TreeExpr) -> int:
    """Return the number of internal eml nodes."""

    if tree in {"x", "1"}:
        return 0

    left, right = tree
    return 1 + tree_internal_node_count(left) + tree_internal_node_count(right)


def tree_shape_summary(tree: TreeExpr) -> TreeShapeSummary:
    """Return structural summary values for one tree."""

    internal = tree_internal_node_count(tree)
    leaves = tree_leaf_count(tree)
    return TreeShapeSummary(
        height=tree_height(tree),
        leaf_count=leaves,
        internal_node_count=internal,
        total_node_count=internal + leaves,
    )


def tree_to_sexpr(tree: TreeExpr) -> str:
    """Serialize a tree to the abbreviated s-expression form."""

    if tree in {"x", "1"}:
        return tree

    left, right = tree
    return f"({tree_to_sexpr(left)} {tree_to_sexpr(right)})"


def tree_to_explicit_eml_sexpr(tree: TreeExpr) -> str:
    """Serialize a tree to an explicit eml(...) s-expression."""

    if tree in {"x", "1"}:
        return tree

    left, right = tree
    return f"(eml {tree_to_explicit_eml_sexpr(left)} {tree_to_explicit_eml_sexpr(right)})"


def exact_height_leaf_trees() -> tuple[TreeExpr, ...]:
    """Return the exact-height zero trees."""

    return ("x", "1")


def iter_exact_height_trees(
    height: int,
    exact_by_height: dict[int, tuple[TreeExpr, ...]],
) -> Iterator[TreeExpr]:
    """Yield all trees of exact height using a cache of smaller heights."""

    if height < 0:
        raise ValueError("height must be non-negative")
    if height == 0:
        yield from exact_height_leaf_trees()
        return

    previous_exact = exact_by_height[height - 1]
    previous_exact_set = set(previous_exact)
    lower_or_equal = []
    for current_height in range(height):
        lower_or_equal.extend(exact_by_height[current_height])

    for left, right in product(lower_or_equal, repeat=2):
        if left in previous_exact_set or right in previous_exact_set:
            yield (left, right)


def build_exact_tree_cache(max_height: int) -> dict[int, tuple[TreeExpr, ...]]:
    """Build an exact-height cache up to max_height."""

    if max_height < 0:
        raise ValueError("max_height must be non-negative")

    cache: dict[int, tuple[TreeExpr, ...]] = {0: exact_height_leaf_trees()}
    for height in range(1, max_height + 1):
        cache[height] = tuple(iter_exact_height_trees(height, cache))
    return cache
