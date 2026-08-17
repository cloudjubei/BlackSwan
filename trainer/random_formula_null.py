"""The matched-complexity RANDOM-FORMULA NULL -- the honesty gauntlet's centrepiece and the direct answer to the
formulaic-alpha-mining wave (AlphaGen/AlphaForge/AlphaAgent/Chain-of-Alpha): a mined alpha only counts as a
discovery if it beats RANDOM formulas of the SAME structural complexity, evaluated the same way, net of cost. We
build random reverse-Polish expression trees of a fixed depth over a supplied feature panel, turn each into a
dollar-neutral cross-sectional book, LAG it one period (so the signal at t-1 trades period t -- no look-ahead),
charge turnover cost, and read the net annualised Sharpe. K such formulas form an empirical null; a claimed
strategy's Sharpe is scored against it with a continuity-corrected empirical p-value. The null is centred at zero
by construction (dollar-neutral + lagged), so any real edge must clear the luck of matched-complexity search.

Complexity is the tree DEPTH; matching it to a paper's operator-vocabulary depth is what makes the null 'matched'.
Signals operate on a panel of (T, N) feature arrays -- T periods, N assets -- the canonical cross-sectional
alpha-mining setting. numpy only; deterministic given the passed rng."""
import numpy as np

from trainer.trading_costs import net_return_series

_EPS = 1e-12


def signal_to_weights(signal):
    """Cross-sectional dollar-neutral book from a raw signal: demean each row, then L1-normalise to unit gross
    exposure. A row with no cross-sectional dispersion (all equal) becomes cash (zero)."""
    s = np.nan_to_num(np.asarray(signal, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if s.ndim != 2:
        raise ValueError("signal must be 2-D (T, N)")
    demeaned = s - s.mean(axis=1, keepdims=True)
    gross = np.abs(demeaned).sum(axis=1, keepdims=True)
    return np.divide(demeaned, gross, out=np.zeros_like(demeaned), where=gross > _EPS)


def _cs_rank(x):
    order = np.argsort(np.argsort(x, axis=1), axis=1).astype(float)
    n = x.shape[1]
    return order / (n - 1) - 0.5 if n > 1 else np.zeros_like(x)


def _delay(x):
    return np.vstack([np.zeros((1, x.shape[1])), x[:-1]])


def _ts_mean(x, k=3):
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        out[i] = x[max(0, i - k + 1):i + 1].mean(axis=0)
    return out


def _cs_zscore(x):
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return np.divide(x - mu, sd, out=np.zeros_like(x), where=sd > _EPS)


_UNARY = [
    lambda x: -x,
    np.sign,
    np.abs,
    _cs_rank,
    _delay,
    _ts_mean,
    _cs_zscore,
]


def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=np.abs(b) > _EPS)


_BINARY = [
    lambda a, b: a + b,
    lambda a, b: a - b,
    lambda a, b: a * b,
    _safe_div,
]


def random_formula(rng, features, depth):
    """Evaluate a random expression of the given DEPTH over the feature panel, returning a (T, N) signal. Depth 0
    picks a base feature; deeper nodes pick a random unary or binary operator and recurse. Deterministic given
    `rng`."""
    feats = [np.nan_to_num(np.asarray(f, dtype=float), nan=0.0, posinf=0.0, neginf=0.0) for f in features]
    if not feats:
        raise ValueError("features must be non-empty")

    def build(d):
        if d <= 0 or rng.random() < 0.15:
            return feats[rng.integers(len(feats))].copy()
        if rng.random() < 0.5:
            op = _UNARY[rng.integers(len(_UNARY))]
            return np.nan_to_num(op(build(d - 1)), nan=0.0, posinf=0.0, neginf=0.0)
        op = _BINARY[rng.integers(len(_BINARY))]
        return np.nan_to_num(op(build(d - 1), build(d - 1)), nan=0.0, posinf=0.0, neginf=0.0)

    return build(depth)


def formula_sharpe(signal, asset_returns, fee=0.0, periods_per_year=252.0):
    """Net-of-cost annualised Sharpe of a signal traded as a dollar-neutral cross-sectional book. The book is
    LAGGED one period (weights from signal[t-1] earn asset_returns[t]) so no look-ahead is possible."""
    weights = signal_to_weights(signal)
    lagged = np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]]) if weights.shape[0] else weights
    net = net_return_series(lagged, np.asarray(asset_returns, dtype=float), fee)
    if net.shape[0] < 2:
        return 0.0
    sd = net.std(ddof=1)
    if not np.isfinite(sd) or sd <= _EPS:
        return 0.0
    return float(net.mean() / sd * np.sqrt(periods_per_year))


def random_formula_null(rng, features, asset_returns, depth, k, fee=0.0, periods_per_year=252.0):
    """K net-of-cost annualised Sharpes from K random depth-`depth` formulas -- the empirical null distribution of
    matched-complexity luck."""
    r = np.asarray(asset_returns, dtype=float)
    out = np.empty(int(k), dtype=float)
    for i in range(int(k)):
        out[i] = formula_sharpe(random_formula(rng, features, depth), r, fee, periods_per_year)
    return out


def null_pvalue(observed, null_sharpes):
    """Continuity-corrected empirical p-value: the fraction of null formulas whose Sharpe is at least the observed
    one, (1 + #{null >= observed}) / (K + 1). Small = the claim beats matched-complexity random search."""
    null = np.asarray(null_sharpes, dtype=float)
    return float((1 + int(np.sum(null >= observed))) / (null.size + 1))
