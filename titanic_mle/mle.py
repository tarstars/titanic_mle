"""Definitions for the project-specific mle function."""

from __future__ import annotations

import math


def mle(x: float, y: float) -> float:
    """Compute the project-specific mle value.

    This project defines mle(x, y) as exp(x) - ln(y), not as maximum
    likelihood estimation. The logarithm requires y > 0.
    """

    if y <= 0:
        raise ValueError("mle(x, y) requires y > 0 because ln(y) is undefined otherwise")

    return math.exp(x) - math.log(y)
