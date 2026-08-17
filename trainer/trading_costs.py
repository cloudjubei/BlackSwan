"""Canonical net-of-cost primitives for the honesty gauntlet: convert a position book (weights over assets per
period) into a gross and then a NET per-period return series, charging a proportional fee on turnover exactly as
the individual strategy modules (tsmom, xsection, globalfactors) do inline -- turnover is the summed absolute
weight change (both legs), the first period charges entry from cash, and cost is fee x turnover. Centralising it
here means the gauntlet scores every claimed strategy on the same net-of-cost convention rather than trusting each
paper's own (often gross) accounting. Contemporaneous convention: weights[t] are the positions held into period t
and must already be point-in-time (lagged by the caller); no look-ahead is introduced here. numpy only."""
import numpy as np


def turnover_series(weights):
    """Per-period turnover = sum of absolute weight changes across assets, with the first period charged the full
    entry from a zero (cash) book. Shape (T, N) -> (T,)."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2:
        raise ValueError("weights must be 2-D (T, N)")
    if w.shape[0] == 0:
        return np.zeros(0, dtype=float)
    prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    return np.abs(w - prev).sum(axis=1)


def portfolio_gross_returns(weights, asset_returns):
    """Gross per-period portfolio return = sum_i weights[t, i] * asset_returns[t, i]. Both (T, N) -> (T,)."""
    w = np.asarray(weights, dtype=float)
    r = np.asarray(asset_returns, dtype=float)
    if w.shape != r.shape:
        raise ValueError("weights and asset_returns must share shape")
    if w.shape[0] == 0:
        return np.zeros(0, dtype=float)
    return (w * r).sum(axis=1)


def net_return_series(weights, asset_returns, fee):
    """Net per-period return = gross - fee * turnover. `fee` is the proportional cost per unit turnover (e.g.
    0.001 = 10 bps). Zero-turnover periods are unaffected by the fee; a full flip costs 2 x fee."""
    gross = portfolio_gross_returns(weights, asset_returns)
    if gross.shape[0] == 0:
        return gross
    return gross - float(fee) * turnover_series(weights)
