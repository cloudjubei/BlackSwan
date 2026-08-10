"""Shared past-only extreme-signal core for the non-price probes.

A probe that reads a continuous non-price stream (perp funding, taker imbalance) reduces each decision bar to
one scalar and asks: is that scalar extreme versus its OWN PAST? An extreme on the high side is one crowd, on
the low side the other; a value in the middle is no signal. This is the exact rule the funding and order-flow
probes share, so it lives here once — the one place the past-only-versus-whole-sample lookahead can hide.
Pure (numpy only), so it stays trivially testable and carries no data-loading or backtest dependency.
"""

import numpy as np


def _is_num(x):
    return isinstance(x, (int, float)) and np.isfinite(x)


def past_only_extreme_sides(values, invert, pct, min_history):
    """Per-bar entry side (+1 / -1 / ``None``) from each bar's scalar, judged against its PAST-ONLY distribution.

    For bar t the scalar ``values[t]`` is compared against the distribution of the finite scalars observed at
    bars 0..t-1 — strictly before t, so the current value is never folded into its own reference set and a
    value that is extreme versus the whole sample but ordinary versus its own past is left flat (the silent
    lookahead this function exists to prevent). At/above the ``pct`` quantile of that past is a high extreme
    (raw +1); at/below the ``1 - pct`` quantile is a low extreme (raw -1); anything between is ``None``.
    ``invert`` negates the side (the contrarian reading — fade the crowd instead of riding it). A bar whose
    value is ``None``/non-finite, or one with fewer than ``min_history`` prior finite observations, is ``None``
    and does not itself enter the distribution."""
    pct = float(pct)
    n = len(values)
    sides = [None] * n
    history = []  # finite scalars observed at bars strictly before the current one
    for t in range(n):
        v = values[t]
        if _is_num(v) and len(history) >= int(min_history):
            hi = float(np.quantile(history, pct))
            lo = float(np.quantile(history, 1.0 - pct))
            raw = 1.0 if v >= hi else (-1.0 if v <= lo else 0.0)
            if raw != 0.0:
                sides[t] = (-raw if invert else raw)
        if _is_num(v):
            history.append(float(v))
    return sides
