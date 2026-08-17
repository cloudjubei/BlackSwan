"""Pure mutual-information estimators (nats) for the METHOD-bet MI-complexity law: each prediction target gets an
MI budget MI(features; target), and the law (complexity's net-of-cost value is governed by MI headroom, with a
threshold MI*) must be shown robust to the estimator. Four complementary estimators:
  gaussian  — closed form from Pearson rho (-0.5*ln(1-rho^2)); exact for jointly-normal, linear-only otherwise;
  copula    — rank-transform both margins to normal scores then gaussian; invariant to monotone transforms;
  binning   — equal-frequency histogram plug-in; captures nonlinear dependence, positively biased at small n;
  ksg       — Kraskov-Stogbauer-Grassberger k-NN (via sklearn mutual_info_regression); continuous, low-bias.
All clip at 0 (MI >= 0) and degrade to 0.0 on too-short input rather than raising. numpy + scipy + sklearn only."""
import math

import numpy as np
from scipy.stats import norm
from sklearn.feature_selection import mutual_info_regression


def _clean_pair(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def gaussian_mi(x, y):
    x, y = _clean_pair(x, y)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return 0.0
    rho = float(np.corrcoef(x, y)[0, 1])
    rho = max(min(rho, 1.0 - 1e-12), -1.0 + 1e-12)
    return float(max(0.0, -0.5 * math.log(1.0 - rho ** 2)))


def _normal_scores(v):
    n = v.size
    ranks = np.argsort(np.argsort(v))
    u = (ranks + 0.5) / n
    return norm.ppf(u)


def gaussian_copula_mi(x, y):
    x, y = _clean_pair(x, y)
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return 0.0
    return gaussian_mi(_normal_scores(x), _normal_scores(y))


def _auto_bins(n):
    return max(2, min(20, int(round(math.sqrt(n / 5.0)))))


def _equal_freq_labels(v, bins):
    edges = np.quantile(v, np.linspace(0.0, 1.0, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return np.clip(np.digitize(v, edges[1:-1]), 0, bins - 1)


def binning_mi(x, y, bins=None):
    x, y = _clean_pair(x, y)
    n = x.size
    if n < 6 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return 0.0
    b = _auto_bins(n) if bins is None else int(bins)
    xl, yl = _equal_freq_labels(x, b), _equal_freq_labels(y, b)
    joint = np.zeros((b, b), dtype=float)
    for i, j in zip(xl, yl):
        joint[i, j] += 1.0
    joint /= n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = joint > 0
    return float(max(0.0, np.sum(joint[mask] * np.log(joint[mask] / denom[mask]))))


def ksg_mi(x, y, k=3):
    x, y = _clean_pair(x, y)
    if x.size < k + 2 or x.std() == 0 or y.std() == 0:
        return 0.0
    val = mutual_info_regression(x.reshape(-1, 1), y, n_neighbors=k, random_state=0)[0]
    return float(max(0.0, val))


def estimate_mi(x, y, k=3, bins=None):
    return {"gaussian": gaussian_mi(x, y), "copula": gaussian_copula_mi(x, y),
            "binning": binning_mi(x, y, bins=bins), "ksg": ksg_mi(x, y, k=k)}
