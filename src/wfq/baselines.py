"""Fixed-weight baselines for WFQ."""


def equal_weights(n_queues):
    """Return equal bandwidth weights for all queues."""

    if not isinstance(n_queues, int) or isinstance(n_queues, bool):
        raise ValueError("n_queues must be a positive integer")

    if n_queues <= 0:
        raise ValueError("n_queues must be positive")

    weight = 1.0 / n_queues

    return [weight] * n_queues


def proportional_weights(arrival_rates):
    """Return bandwidth weights proportional to arrival rates."""

    rates = list(arrival_rates)

    if not rates:
        raise ValueError("arrival_rates cannot be empty")

    for rate in rates:
        if rate <= 0:
            raise ValueError("arrival rates must be positive")

    total_rate = sum(rates)

    return [rate / total_rate for rate in rates]