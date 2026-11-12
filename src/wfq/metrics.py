"""Evaluation metrics for the WFQ project."""

import numpy as np


def jains_index(allocations):
    """Calculate Jain's fairness index for a list of positive values.

    Jain's index = (sum(x_i))^2 / (n * sum(x_i^2))
    Result is between 1/n (worst fairness) and 1.0 (perfect fairness).
    If all values are zero, returns 1.0 (equal treatment).
    """

    values = [float(v) for v in allocations]

    if not values:
        raise ValueError("allocations cannot be empty")

    for v in values:
        if v < 0.0:
            raise ValueError("allocations must be non-negative")

    total = sum(values)

    if total == 0.0:
        return 1.0

    sum_of_squares = sum(v * v for v in values)
    n = len(values)

    return (total * total) / (n * sum_of_squares)
