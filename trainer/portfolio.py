"""Diversified-portfolio / breadth analysis — PURE (no torch, no filesystem, no model).

Combines per-asset strategy equity curves (each a RunSummary `series.equity`) into ONE portfolio equity curve
and reports basis-free risk/return stats, so a breadth experiment (e.g. diversified trend across low-correlation
markets) is a RE-MEASURABLE, testable component instead of an ad-hoc script. Consumers pass equity curves
(lists/arrays); this module never reads disk or runs anything. Curves for the SAME walk-forward window share a
downsampling, so they align index-wise; combine() truncates to the shortest to be safe.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np


def step_returns(equity: Sequence[float]) -> np.ndarray:
    """Per-step simple returns of an equity curve; non-finite steps (0/0 etc.) neutralised to 0."""
    eq = np.asarray(equity, dtype=float)
    if eq.size < 2:
        return np.zeros(0, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = eq[1:] / eq[:-1] - 1.0
    return np.where(np.isfinite(r), r, 0.0)


def curve_stats(equity: Sequence[float]) -> Dict[str, float]:
    """Basis-free stats for one equity curve: total return %, max drawdown %, per-step Sharpe (mean/std of
    step returns), and Calmar (return / |maxDD|). Sharpe is per-step (unannualised) — a RELATIVE measure
    valid for comparing curves computed the same way; annualised/deflated Sharpe is the engine's verdict layer."""
    eq = np.asarray(equity, dtype=float)
    if eq.size < 2 or eq[0] <= 0:
        return {"total_return_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe": 0.0, "calmar": 0.0}
    r = step_returns(eq)
    total = (eq[-1] / eq[0] - 1.0) * 100.0
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min() * 100.0)
    sharpe = float(r.mean() / r.std()) if r.std() > 1e-12 else 0.0
    # Calmar = return per unit of drawdown. With NO drawdown it is undefined (return/0) — return a neutral 0
    # rather than inf/nan so a flat/degenerate curve never spuriously ranks best (real curves always draw down).
    calmar = (total / abs(maxdd)) if maxdd < -1e-9 else 0.0
    return {"total_return_pct": total, "max_drawdown_pct": maxdd, "sharpe": sharpe, "calmar": calmar}


def inverse_vol_weights(curves: List[Sequence[float]]) -> np.ndarray:
    """Risk-parity weights: each curve weighted by the inverse of its step-return volatility (higher-vol legs
    contribute less), normalised to sum 1. Zero-vol curves get zero weight; all-zero falls back to equal."""
    vols = np.array([step_returns(c).std() for c in curves], dtype=float)
    inv = np.where(vols > 1e-12, 1.0 / vols, 0.0)
    return (inv / inv.sum()) if inv.sum() > 0 else np.full(len(curves), 1.0 / max(1, len(curves)))


def combine(curves: List[Sequence[float]], weights: Optional[Sequence[float]] = None,
            initial: float = 100000.0) -> np.ndarray:
    """Combine per-asset equity curves into ONE portfolio equity curve by weighting their per-step RETURNS
    (daily-rebalanced to `weights`; equal-weight if None). Curves are truncated to the shortest so same-window
    curves align. Empty/too-short input returns a flat one-point curve at `initial`."""
    rets = [step_returns(c) for c in curves if np.asarray(c, dtype=float).size >= 2]
    if not rets:
        return np.asarray([initial], dtype=float)
    n = min(len(r) for r in rets)
    if n < 1:
        return np.asarray([initial], dtype=float)
    R = np.array([r[:n] for r in rets])
    w = np.full(len(R), 1.0 / len(R)) if weights is None else np.asarray(weights, dtype=float)[: len(R)]
    w = w / w.sum() if w.sum() != 0 else np.full(len(R), 1.0 / len(R))
    rp = (R * w[:, None]).sum(axis=0)
    return np.concatenate([[initial], initial * np.cumprod(1.0 + rp)])


def diversified_stats(curves: List[Sequence[float]], weights: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Stats of the combined portfolio (equal-weight unless weights given). Pass inverse_vol_weights(curves)
    for a risk-parity basket."""
    return curve_stats(combine(curves, weights))
