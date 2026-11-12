"""Tests for evaluation metrics."""

import pytest

from wfq.metrics import jains_index


def test_jains_index_equal_values():
    # Equal values must have perfect fairness (1.0).

    assert jains_index([10.0, 10.0, 10.0, 10.0]) == pytest.approx(1.0)
    assert jains_index([1.0, 1.0]) == pytest.approx(1.0)


def test_jains_index_worst_case():
    # One non-zero value among N must equal 1/N.

    n = 8
    allocations = [0.0] * n
    allocations[0] = 5.0

    assert jains_index(allocations) == pytest.approx(1.0 / n)


def test_jains_index_all_zeros():
    # All zeros treat every queue equally.

    assert jains_index([0.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_jains_index_validation():
    # Negative values and empty lists must be rejected.

    with pytest.raises(ValueError):
        jains_index([])

    with pytest.raises(ValueError):
        jains_index([-1.0, 2.0])
