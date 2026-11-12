"""Tests for Step 5: Equal and Proportional baselines."""

from pathlib import Path

import pytest
import yaml

from wfq.baselines import equal_weights, proportional_weights


# Load project configuration
ROOT = Path(__file__).resolve().parents[1]

CONFIG = yaml.safe_load(
    (ROOT / "config" / "base.yaml").read_text(encoding="utf-8")
)

SIM = CONFIG["simulator"]

RATES = SIM["arrival_rates"]
N = SIM["n_queues"]
BANDWIDTH = SIM["total_bandwidth"]


def test_equal_weights():
    # Equal allocation gives every queue the same bandwidth share.

    weights = equal_weights(N)

    assert len(weights) == N
    assert weights == pytest.approx([1.0 / N] * N)
    assert sum(weights) == pytest.approx(1.0)


def test_proportional_weights_sum_to_one():
    # Proportional allocation must distribute the complete bandwidth.

    weights = proportional_weights(RATES)

    assert sum(weights) == pytest.approx(1.0)


def test_proportional_weights_match_arrival_rates():
    # Each queue's weight must equal its arrival rate divided by total arrivals.

    weights = proportional_weights(RATES)

    total_rate = sum(RATES)

    expected = [
        rate / total_rate
        for rate in RATES
    ]

    assert weights == pytest.approx(expected)


def test_proportional_weights_preserve_arrival_rate_ratios():
    # The ratio between any two weights must match their arrival rate ratio.

    weights = proportional_weights(RATES)

    for i in range(N):
        for j in range(N):
            assert weights[i] / weights[j] == pytest.approx(
                RATES[i] / RATES[j]
            )


def test_proportional_allocation_gives_equal_service_load():
    # Every queue should have the same arrival-to-service-rate ratio.

    weights = proportional_weights(RATES)

    total_rate = sum(RATES)

    expected_load = total_rate / BANDWIDTH

    for rate, weight in zip(RATES, weights):

        service_rate = weight * BANDWIDTH

        load = rate / service_rate

        assert load == pytest.approx(expected_load)


def test_invalid_rates_refused():
    # Validation tests for zero, negative, or empty inputs.

    with pytest.raises(ValueError):
        proportional_weights([])

    with pytest.raises(ValueError):
        proportional_weights([-0.1, 0.2, 0.3])

    with pytest.raises(ValueError):
        proportional_weights([0.0, 0.2, 0.3])

    with pytest.raises(ValueError):
        equal_weights(0)

    with pytest.raises(ValueError):
        equal_weights(-5)
