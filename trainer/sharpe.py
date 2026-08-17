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


# --- powered-null primitives -------------------------------------------------------------------------------
# A null is only informative if it is POWERED: "we found nothing at t>3" is indistinguishable from "we ran a
# test with no power" until you report the smallest effect the sample could have detected and a confidence
# interval on the true Sharpe. These reuse the same non-normality-adjusted variance term as the PSR (the Lo /
# Mertens standard error), so the Sharpe SE, the minimum detectable effect, and the DSR all speak one geometry.
# All Sharpes are per-observation (NOT annualised) — the caller annualises for reporting and passes `sr_econ`
# in the same per-observation units.


def _psr_denominator(sharpe, skewness, kurt):
    return 1.0 - float(skewness) * float(sharpe) + ((float(kurt) - 1.0) / 4.0) * float(sharpe) ** 2


def sharpe_standard_error(sharpe, skewness, kurt, n_obs):
    """Non-normality-adjusted standard error of the per-observation Sharpe (Lo 2002 / Mertens):
    sqrt(denom / (n-1)) with denom = 1 - g3*SR + (g4-1)/4*SR^2 (the PSR denominator; `kurt` NON-excess).
    inf when undefined (n < 2 or a non-positive denominator)."""
    n = int(n_obs)
    if n < 2:
        return math.inf
    denom = _psr_denominator(sharpe, skewness, kurt)
    if denom <= 0 or not np.isfinite(denom):
        return math.inf
    return float(math.sqrt(denom / (n - 1)))


def sharpe_confidence_interval(sharpe, skewness, kurt, n_obs, alpha=0.05):
    """Two-sided (1-alpha) confidence interval for the true per-observation Sharpe, using the Lo SE.
    (-inf, inf) when the SE is undefined."""
    se = sharpe_standard_error(sharpe, skewness, kurt, n_obs)
    if not np.isfinite(se):
        return (-math.inf, math.inf)
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return (float(sharpe) - z * se, float(sharpe) + z * se)


def sharpe_power(sr_alt, n_obs, alpha=0.05, skewness=0.0, kurt=3.0):
    """Power of the one-sided level-alpha test H0: SR<=0 to detect a true per-observation Sharpe `sr_alt`
    at `n_obs` observations: Phi( sr_alt*sqrt(n)/sqrt(denom(sr_alt)) - z_{1-alpha} ). 0.0 when undefined."""
    n = int(n_obs)
    if n < 2:
        return 0.0
    denom = _psr_denominator(sr_alt, skewness, kurt)
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    z_a = float(norm.ppf(1.0 - alpha))
    return float(norm.cdf(float(sr_alt) * math.sqrt(n) / math.sqrt(denom) - z_a))


def minimum_detectable_sharpe(n_obs, alpha=0.05, power=0.8, skewness=0.0, kurt=3.0):
    """Smallest true per-observation Sharpe a one-sided level-alpha test detects at `power` given `n_obs`.
    Solves sr = (z_{1-alpha} + z_{power}) * sqrt(denom(sr)/n); iterated because the moment-adjusted denom
    depends on sr (seeded at the normal denom=1 solution). inf for n < 2."""
    n = int(n_obs)
    if n < 2:
        return math.inf
    za, zb = float(norm.ppf(1.0 - alpha)), float(norm.ppf(power))
    sr = (za + zb) / math.sqrt(n)
    for _ in range(64):
        denom = _psr_denominator(sr, skewness, kurt)
        if denom <= 0 or not np.isfinite(denom):
            break
        nxt = (za + zb) * math.sqrt(denom / n)
        if abs(nxt - sr) < 1e-13:
            sr = nxt
            break
        sr = nxt
    return float(sr)


def benjamini_hochberg(pvalues, q=0.05):
    """Benjamini-Hochberg FDR control at level q. Returns a boolean list aligned to `pvalues` (True = reject).
    Step-up: reject the largest-k ordered p with p_(k) <= (k/m)*q, and everything ranked below it."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p, kind="mergesort")
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        kmax = int(np.max(np.where(passed)[0]))
        reject[order[: kmax + 1]] = True
    return reject.tolist()


def newey_west_inflation(autocorrs, q):
    """Lo (2002) / Newey-West variance-inflation factor for the Sharpe SE under serial correlation:
    eta = 1 + 2*sum_{k=1..q} (1 - k/(q+1)) * rho_k, `autocorrs` = [rho_1, rho_2, ...] (only the first q used,
    Bartlett-weighted). Floored at a small positive so a strongly negatively-autocorrelated series can shrink
    but never invert the variance. eta = 1 (no adjustment) for q < 1 or no autocorrelations."""
    q = int(q)
    if q < 1:
        return 1.0
    ac = list(autocorrs)[:q]
    s = sum((1.0 - k / (q + 1.0)) * float(rho) for k, rho in enumerate(ac, start=1))
    return float(max(1.0 + 2.0 * s, 1e-6))


def _autocorrelations(returns, q):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 3:
        return []
    r = r - r.mean()
    denom = float(r @ r)
    if denom <= 0:
        return []
    return [float(r[k:] @ r[:-k]) / denom if k < n else 0.0 for k in range(1, q + 1)]


def _default_hac_lag(n):
    """The standard automatic Bartlett bandwidth q = floor(4 * (n/100)^(2/9))."""
    return max(1, int(math.floor(4.0 * (max(int(n), 1) / 100.0) ** (2.0 / 9.0))))


def sharpe_standard_error_hac(returns, q=None):
    """Serial-correlation-adjusted (Lo 2002) SE of the per-observation Sharpe from a RAW return series: the
    i.i.d. Lo/Mertens SE times sqrt(Newey-West inflation) at Bartlett lag `q` (default 4*(n/100)^(2/9)).
    inf when the i.i.d. SE is undefined."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 2:
        return math.inf
    st = sharpe_stats(r)
    se_iid = sharpe_standard_error(st["sharpe"], st["skew"], st["kurtosis"], st["n_obs"])
    if not np.isfinite(se_iid):
        return math.inf
    q = _default_hac_lag(n) if q is None else int(q)
    eta = newey_west_inflation(_autocorrelations(r, q), q)
    return float(se_iid * math.sqrt(eta))


def probabilistic_sharpe_ratio_hac(returns, sr_benchmark=0.0, q=None):
    """PSR computed with the Lo-2002 serial-correlation-robust (HAC) Sharpe SE instead of the i.i.d. SE the
    standard PSR uses. The Bailey-Lopez de Prado PSR adjusts for skew/kurtosis (Mertens) but OMITS the
    autocorrelation correction, so for serially-dependent PnL (smoothed / illiquid / higher-frequency) it
    understates the Sharpe SE and is ANTI-CONSERVATIVE (fires too early). This variant deflates by
    sqrt(Newey-West inflation): PSR_hac = Phi( (SR - SR*) / se_hac ). 0.0 when the SE is undefined (n < 3 or a
    non-positive/degenerate denominator)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return 0.0
    st = sharpe_stats(r)
    se = sharpe_standard_error_hac(r, q=q)
    if not np.isfinite(se) or se <= 0:
        return 0.0
    return float(norm.cdf((st["sharpe"] - float(sr_benchmark)) / se))


def benjamini_yekutieli(pvalues, q=0.05):
    """Benjamini-Yekutieli FDR under ARBITRARY dependence: Benjamini-Hochberg with q scaled by 1/H_m, where
    H_m = sum_{i=1..m} 1/i. Strictly more conservative than BH; the honest bound when tests are dependent."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return []
    hm = float(np.sum(1.0 / np.arange(1, m + 1)))
    return benjamini_hochberg(pvalues, q=q / hm)


def powered_null_verdict(sharpe, skewness, kurt, n_obs, sr_econ, alpha=0.05, power=0.8):
    """Per-cell POWERED verdict (per-observation Sharpe units). Distinguishes a real null from a silent lack
    of power using one-sided (1-alpha) bounds on the true Sharpe:
      survivor      — one-sided LOWER bound > 0 (the effect is significantly positive);
      powered-null  — one-sided UPPER bound < `sr_econ` (we can REJECT a true Sharpe >= the economically
                      meaningful `sr_econ`), i.e. an earned "no edge";
      inconclusive  — neither: the sample cannot rule out `sr_econ`, so absence of evidence is not evidence.
    Also returns the SE, the two one-sided bounds, the minimum detectable Sharpe, and the power at `sr_econ`."""
    se = sharpe_standard_error(sharpe, skewness, kurt, n_obs)
    mde = minimum_detectable_sharpe(n_obs, alpha=alpha, power=power, skewness=skewness, kurt=kurt)
    pwr = sharpe_power(sr_econ, n_obs, alpha=alpha, skewness=skewness, kurt=kurt)
    if not np.isfinite(se):
        return {"verdict": "inconclusive", "se": se, "lower_bound": -math.inf,
                "upper_bound": math.inf, "mde": mde, "power_at_econ": pwr}
    za = float(norm.ppf(1.0 - alpha))
    lower = float(sharpe) - za * se
    upper = float(sharpe) + za * se
    if lower > 0:
        verdict = "survivor"
    elif upper < float(sr_econ):
        verdict = "powered-null"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "se": se, "lower_bound": lower, "upper_bound": upper,
            "mde": mde, "power_at_econ": pwr}
