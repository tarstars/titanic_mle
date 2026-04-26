"""Count valid two-input EML trees on a finite discrete domain.

The grammar is:

    T := 1 | x0 | x1 | (T T)

where ``(A B)`` denotes one internal ``eml(A, B)`` node.

This module is intentionally specialized for the current Titanic experiment:
we want exact counts for trees of height ``<= 5`` that can be evaluated on
every pair from the discrete ``sex x embarked`` domain.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


DomainPoint = tuple[float, float]
ValueSignature = tuple[float, ...]
SignatureCounter = Counter[ValueSignature]

SEX_EMBARKED_DOMAIN: tuple[DomainPoint, ...] = (
    (0.0, 0.0),
    (0.0, 0.5),
    (0.0, 1.0),
    (1.0, 0.0),
    (1.0, 0.5),
    (1.0, 1.0),
)


@dataclass(frozen=True)
class TwoInputValidTreeCountSummary:
    """Exact counting summary for two-input valid trees."""

    domain: tuple[DomainPoint, ...]
    exact_height_counts: dict[int, int]
    unique_signature_counts: dict[int, int]
    left_safe_counts: dict[int, int]
    right_safe_counts: dict[int, int]
    total_count_le_5: int


def _leaf_signature_counter(domain: tuple[DomainPoint, ...]) -> SignatureCounter:
    """Return the exact-height-zero signature multiset."""

    one = tuple(1.0 for _ in domain)
    x0 = tuple(point[0] for point in domain)
    x1 = tuple(point[1] for point in domain)
    return Counter({one: 1, x0: 1, x1: 1})


def _is_left_safe(values: ValueSignature) -> bool:
    """Return whether a signature is safe on the exp branch."""

    try:
        return all(math.isfinite(math.exp(value)) for value in values)
    except OverflowError:
        return False


def _is_right_safe(values: ValueSignature) -> bool:
    """Return whether a signature is safe on the log branch."""

    try:
        return all(value > 0.0 and math.isfinite(math.log(value)) for value in values)
    except ValueError:
        return False


def _build_exact_signature_counters(
    max_height: int,
    domain: tuple[DomainPoint, ...],
) -> tuple[SignatureCounter, ...]:
    """Return exact-height signature counters up to ``max_height`` inclusive."""

    if max_height < 0:
        raise ValueError("max_height must be non-negative")

    exact: list[SignatureCounter] = [_leaf_signature_counter(domain)]
    for height in range(1, max_height + 1):
        previous_exact_keys = set(exact[height - 1])

        lower: SignatureCounter = Counter()
        for lower_height in range(height):
            lower.update(exact[lower_height])

        lower_items = tuple(lower.items())
        current: SignatureCounter = Counter()
        for left_values, left_count in lower_items:
            left_is_previous = left_values in previous_exact_keys
            for right_values, right_count in lower_items:
                if not (left_is_previous or right_values in previous_exact_keys):
                    continue

                try:
                    values = tuple(
                        math.exp(left) - math.log(right)
                        for left, right in zip(left_values, right_values)
                    )
                except (OverflowError, ValueError):
                    continue

                if all(math.isfinite(value) for value in values):
                    current[values] += left_count * right_count

        exact.append(current)

    return tuple(exact)


def count_valid_two_input_trees_height_le_five(
    domain: tuple[DomainPoint, ...] = SEX_EMBARKED_DOMAIN,
) -> TwoInputValidTreeCountSummary:
    """Count valid trees of height ``<= 5`` on the given discrete domain.

    The expensive part is building the exact signature counters through height 4.
    For exact height 5 we only need counts, not signatures:

    - a left subtree is admissible iff it is valid and safe for ``exp``
    - a right subtree is admissible iff it is valid and safe for ``ln``

    Exact-height-5 trees are counted in two disjoint cases:

    1. left child has exact height 4, right child has height <= 4
    2. left child has height <= 3, right child has exact height 4
    """

    exact = _build_exact_signature_counters(max_height=4, domain=domain)

    exact_height_counts = {
        height: sum(counter.values())
        for height, counter in enumerate(exact)
    }
    unique_signature_counts = {
        height: len(counter)
        for height, counter in enumerate(exact)
    }
    left_safe_counts = {
        height: sum(count for values, count in counter.items() if _is_left_safe(values))
        for height, counter in enumerate(exact)
    }
    right_safe_counts = {
        height: sum(count for values, count in counter.items() if _is_right_safe(values))
        for height, counter in enumerate(exact)
    }

    exact_height_5_count = (
        left_safe_counts[4] * sum(right_safe_counts.values())
        + sum(left_safe_counts[height] for height in range(4)) * right_safe_counts[4]
    )
    exact_height_counts[5] = exact_height_5_count

    total_count_le_5 = sum(exact_height_counts.values())

    return TwoInputValidTreeCountSummary(
        domain=domain,
        exact_height_counts=exact_height_counts,
        unique_signature_counts=unique_signature_counts,
        left_safe_counts=left_safe_counts,
        right_safe_counts=right_safe_counts,
        total_count_le_5=total_count_le_5,
    )
