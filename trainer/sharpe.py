"""Sharpe / Deflated-Sharpe-Ratio primitives (Bailey & Lopez de Prado) — the multiple-testing rigor
gate for the Wave-2 verdict layer. Pure (numpy + scipy, no torch). All Sharpes are per-observation
(NOT annualised) — the verdict layer feeds in a run's per-step test-window returns.

References: Bailey & Lopez de Prado, "The Sharpe Ratio Efficient Frontier" (PSR) and "The Deflated
Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality"."""
import math

import numpy as np
from scipy.stats import norm, skew, kurtosis

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns):
    """Per-observation Sharpe = mean/std (sample std, ddof=1). 0.0 when undefined (<2 points or
    zero variance) — never inf/nan, so it composes safely downstream."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    if np.ptp(r) == 0:  # constant series: variance is undefined; r.std is only float noise (~1e-18)
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(r.mean() / sd)


def _moments(r):
    """Observed Sharpe, skewness, and non-excess kurtosis (normal == 3) of a return series.
    Constant / too-short series have undefined higher moments -> normal-shaped defaults (0, 3)."""
    r = np.asarray(r, dtype=float)
    sr = sharpe_ratio(r)
    if r.size < 2 or np.ptp(r) == 0:
        return sr, 0.0, 3.0
    return sr, float(skew(r, bias=False)), float(kurtosis(r, fisher=False, bias=False))


def sharpe_stats(returns):
    """The per-run bundle the verdict layer's Deflated Sharpe Ratio needs: per-observation Sharpe, the
    return distribution's skew + non-excess kurtosis, and the sample length. Safe on flat/short series."""
    r = np.asarray(returns, dtype=float)
    sr, g3, g4 = _moments(r)
    return {"sharpe": sr, "skew": g3, "kurtosis": g4, "n_obs": int(r.size)}


def psr_from_stats(sharpe, skewness, kurt, n_obs, sr_benchmark=0.0):
    """PSR from a PRECOMPUTED moment bundle — `sharpe`/`skewness`/`kurt`/`n_obs` (exactly what `sharpe_stats`
    emits and summary.py stores per run) — instead of a raw return array. This is the single source of the
    PSR closed form; the array-based `probabilistic_sharpe_ratio` delegates here, and the modeltrainer engine's
    TS Deflated-Sharpe port mirrors it (fed the per-run oos_* metrics). `kurt` is NON-excess (normal == 3),
    matching `_moments`. PSR = Phi( (SR - SR*) * sqrt(n-1) / sqrt(1 - g3*SR + (g4-1)/4 * SR^2) ); 0.0 when
    undefined (n < 2 or a non-positive denominator)."""
    n = int(n_obs)
    if n < 2:
        return 0.0
    sr = float(sharpe)
    denom = 1.0 - float(skewness) * sr + ((float(kurt) - 1.0) / 4.0) * sr * sr
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(norm.cdf(z))


def probabilistic_sharpe_ratio(returns, sr_benchmark=0.0):
    """PSR from a raw return series — computes the moments then delegates to {@link psr_from_stats}."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sr, g3, g4 = _moments(r)
    return psr_from_stats(sr, g3, g4, r.size, sr_benchmark=sr_benchmark)


def expected_max_sharpe(n_trials, trial_sr_std):
    """The deflation level SR* — the expected MAXIMUM Sharpe across `n_trials` independent trials
    under the null (true SR = 0), scaled by the cross-trial Sharpe std. 0.0 for <2 trials (no
    multiple testing) or no spread. SR* = trial_sr_std * [ (1-gamma)*Z^-1(1 - 1/N) + gamma*Z^-1(1 -
    1/(N*e)) ], gamma = Euler-Mascheroni."""
    if n_trials < 2 or trial_sr_std <= 0:
        return 0.0
    g = EULER_MASCHERONI
    bracket = (1.0 - g) * float(norm.ppf(1.0 - 1.0 / n_trials)) + g * float(
        norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )
    return float(trial_sr_std * bracket)


def dsr_from_stats(sharpe, skewness, kurt, n_obs, n_trials, trial_sr_std):
    """DSR from a precomputed moment bundle: PSR (via {@link psr_from_stats}) measured against the
    expected-max-Sharpe deflation level for `n_trials` configs tried. The engine's TS port calls this with
    each run's oos_* metrics + the cross-run `n_trials`/`trial_sr_std` it computes over the sweep."""
    sr_star = expected_max_sharpe(n_trials, trial_sr_std)
    return psr_from_stats(sharpe, skewness, kurt, n_obs, sr_benchmark=sr_star)


def deflated_sharpe_ratio(returns, n_trials, trial_sr_std):
    """DSR from a raw return series: PSR measured against the expected-max-Sharpe deflation level for
    `n_trials` configs tried. DSR > ~0.95 means the observed Sharpe is unlikely to be multiple-testing luck."""
    sr_star = expected_max_sharpe(n_trials, trial_sr_std)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr_star)


def min_track_record_length_from_stats(sharpe, skewness, kurt, n_obs, sr_benchmark=0.0, target_prob=0.95):
    """minTRL from a precomputed moment bundle (see {@link psr_from_stats}) — the min observations for the
    observed Sharpe to clear `sr_benchmark` at `target_prob`. inf when the Sharpe is not above the benchmark."""
    n = int(n_obs)
    if n < 2:
        return math.inf
    sr = float(sharpe)
    if sr <= sr_benchmark:
        return math.inf
    denom = 1.0 - float(skewness) * sr + ((float(kurt) - 1.0) / 4.0) * sr * sr
    if denom <= 0 or not np.isfinite(denom):
        return math.inf
    return 1.0 + denom * (float(norm.ppf(target_prob)) / (sr - sr_benchmark)) ** 2


def min_track_record_length(returns, sr_benchmark=0.0, target_prob=0.95):
    """minTRL from a raw return series — computes the moments then delegates to
    {@link min_track_record_length_from_stats}."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return math.inf
    sr, g3, g4 = _moments(r)
    return min_track_record_length_from_stats(
        sr, g3, g4, r.size, sr_benchmark=sr_benchmark, target_prob=target_prob
    )
