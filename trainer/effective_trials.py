"""Effective number of independent trials for multiplicity deflation. The Deflated Sharpe Ratio deflates by the
expected maximum Sharpe across n INDEPENDENT trials (Bailey-Lopez de Prado), but an automated / LLM search emits
CORRELATED strategies, so plugging the raw trial count over-deflates and false-negatives genuine alpha. This
returns an effective count in [1, M] from the strategies' correlation structure, so the certifier can deflate by
the honest number of independent bets rather than the raw search breadth. Two standard estimators:
  participation - the eigenvalue participation ratio (sum lam)^2 / sum lam^2 of the MxM correlation matrix
                  (independent -> M, one dominant factor -> 1); smooth and monotone in redundancy;
  liji          - Li & Ji (2005): M_eff = sum_i [ 1(lam_i >= 1) + (lam_i - floor(lam_i)) ], exact M for the
                  identity and 1 for a rank-one correlation. numpy only."""
import math

import numpy as np

from trainer.sharpe import expected_max_sharpe


def effective_trials_from_max_sharpe(observed_max_sharpe, trial_sr_std, max_log10_k=15.0):
    """Invert the expected-max-Sharpe deflation to recover the EFFECTIVE number of trials implied by an OBSERVED
    best (spurious) Sharpe -- the continuous / gradient-search analog of counting an enumerated trial set. A model
    fit by gradient descent on a NULL series achieves an in-sample Sharpe `observed_max_sharpe`; the K such that
    expected_max_sharpe(K, trial_sr_std) == observed_max_sharpe is the effective # of independent bets that search
    explored (the effective degrees of freedom), reportable WITHOUT enumerating configs. `trial_sr_std` is the
    per-strategy Sharpe spread of a single fixed strategy on the null (~sqrt(periods_per_year / T)). Floors at 1.0
    (an observation below the 2-trial level, a non-positive spread, or a non-positive observation)."""
    s = float(observed_max_sharpe)
    sd = float(trial_sr_std)
    if sd <= 0 or s <= 0 or s < expected_max_sharpe(2, sd):
        return 1.0
    lo, hi = math.log10(2.0), float(max_log10_k)
    if s >= expected_max_sharpe(10.0 ** hi, sd):
        return 10.0 ** hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if expected_max_sharpe(10.0 ** mid, sd) < s:
            lo = mid
        else:
            hi = mid
    return float(10.0 ** (0.5 * (lo + hi)))


def effective_dof_from_sharpe(observed_annual_sharpe, n_obs, periods_per_year=252.0):
    """Effective DEGREES OF FREEDOM a fit burned to reach an observed in-sample Sharpe on a null series -- the
    robust (bounded, non-exponential) form of the overfit haircut. A p-parameter fit on n null observations has
    in-sample R^2 ~ p/n and annualized Sharpe S ~ sqrt(ppy*R^2/(1-R^2)); inverting, p_eff = n * S^2/(ppy + S^2),
    which recovers p for a linear fit and is bounded by n_obs. The honest deflation level to subtract from a
    reported in-sample Sharpe is then sqrt(ppy * p_eff / n_obs). Preferred over the exp() trial-count form, which
    is p_eff-exponential and saturates. 0 for a non-positive Sharpe."""
    s = float(observed_annual_sharpe)
    n = int(n_obs)
    if s <= 0 or n <= 0:
        return 0.0
    s2 = s * s
    return float(min(n, n * s2 / (periods_per_year + s2)))


def _corr_from_returns(returns_list):
    series = []
    n = min((np.asarray(r, dtype=float).size for r in returns_list), default=0)
    if n < 3:
        return None
    for r in returns_list:
        v = np.asarray(r, dtype=float)[:n]
        v = np.nan_to_num(v - np.nanmean(v))
        s = v.std()
        series.append(v / s if s > 0 else np.zeros(n))
    x = np.column_stack(series)
    return (x.T @ x) / n


def _eigenvalues(corr):
    ev = np.linalg.eigvalsh(np.asarray(corr, dtype=float))
    return ev[ev > 1e-12]


def effective_trials_participation(corr):
    ev = _eigenvalues(corr)
    if ev.size == 0:
        return 1.0
    val = float((ev.sum() ** 2) / (ev ** 2).sum())
    return float(min(max(val, 1.0), corr.shape[0]))


def effective_trials_liji(corr):
    ev = _eigenvalues(corr)
    if ev.size == 0:
        return 1.0
    val = float(np.sum((ev >= 1.0).astype(float) + (ev - np.floor(ev))))
    return float(min(max(val, 1.0), corr.shape[0]))


def effective_trials(returns_list, method="participation"):
    m = len(returns_list)
    if m == 0:
        return 0.0
    if m == 1:
        return 1.0
    corr = _corr_from_returns(returns_list)
    if corr is None:
        return float(m)
    return effective_trials_liji(corr) if method == "liji" else effective_trials_participation(corr)
