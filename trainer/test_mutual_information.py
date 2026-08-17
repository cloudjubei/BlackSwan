"""Unit tests for the pure mutual-information estimators (the METHOD-bet MI-complexity law needs a per-target
MI budget in nats, estimated by SEVERAL estimators so the rung-ordering law can be shown robust to the
estimator). Ground truth: bivariate Gaussian has closed-form MI = -0.5*ln(1-rho^2) nats. numpy + scipy + sklearn
only; no torch."""
import math

import numpy as np
import pytest

from trainer.mutual_information import (
    binning_mi,
    estimate_mi,
    gaussian_copula_mi,
    gaussian_mi,
    ksg_mi,
)


def _bivariate_normal(rho, n, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    x = z[:, 0]
    y = rho * z[:, 0] + math.sqrt(1 - rho ** 2) * z[:, 1]
    return x, y


def _true_gaussian_mi(rho):
    return -0.5 * math.log(1 - rho ** 2)


def test_gaussian_mi_recovers_closed_form():
    x, y = _bivariate_normal(0.8, 200000, seed=1)
    assert gaussian_mi(x, y) == pytest.approx(_true_gaussian_mi(0.8), abs=0.01)


def test_gaussian_mi_zero_on_independent():
    rng = np.random.default_rng(2)
    x, y = rng.standard_normal(100000), rng.standard_normal(100000)
    assert gaussian_mi(x, y) < 0.01


def test_gaussian_mi_monotone_in_rho():
    a = gaussian_mi(*_bivariate_normal(0.1, 50000, seed=3))
    b = gaussian_mi(*_bivariate_normal(0.5, 50000, seed=3))
    c = gaussian_mi(*_bivariate_normal(0.9, 50000, seed=3))
    assert a < b < c


def test_gaussian_mi_never_negative():
    rng = np.random.default_rng(4)
    x, y = rng.standard_normal(500), rng.standard_normal(500)
    assert gaussian_mi(x, y) >= 0.0


def test_binning_mi_independent_near_zero():
    rng = np.random.default_rng(5)
    x, y = rng.standard_normal(20000), rng.standard_normal(20000)
    assert binning_mi(x, y, bins=8) < 0.05


def test_binning_mi_deterministic_recovers_log_bins():
    rng = np.random.default_rng(6)
    x = rng.standard_normal(40000)
    assert binning_mi(x, x.copy(), bins=8) == pytest.approx(math.log(8), abs=0.05)


def test_binning_mi_monotone_in_dependence():
    weak = binning_mi(*_bivariate_normal(0.3, 40000, seed=7), bins=10)
    strong = binning_mi(*_bivariate_normal(0.85, 40000, seed=7), bins=10)
    assert strong > weak > 0.0


def test_ksg_mi_recovers_gaussian():
    x, y = _bivariate_normal(0.7, 4000, seed=8)
    assert ksg_mi(x, y, k=3) == pytest.approx(_true_gaussian_mi(0.7), rel=0.20)


def test_ksg_mi_independent_near_zero():
    rng = np.random.default_rng(9)
    x, y = rng.standard_normal(4000), rng.standard_normal(4000)
    assert ksg_mi(x, y, k=3) < 0.05


def test_copula_mi_invariant_to_monotone_transform():
    x, y = _bivariate_normal(0.6, 40000, seed=10)
    base = gaussian_copula_mi(x, y)
    warped = gaussian_copula_mi(x, np.exp(3.0 * y))
    assert warped == pytest.approx(base, abs=0.02)


def test_gaussian_mi_is_not_transform_invariant_but_copula_is():
    x, y = _bivariate_normal(0.6, 40000, seed=11)
    warped = np.exp(3.0 * y)
    assert abs(gaussian_mi(x, warped) - gaussian_mi(x, y)) > 0.05
    assert abs(gaussian_copula_mi(x, warped) - gaussian_copula_mi(x, y)) < 0.03


def test_estimate_mi_returns_all_estimators_finite_nonneg():
    x, y = _bivariate_normal(0.5, 5000, seed=12)
    out = estimate_mi(x, y)
    assert set(out) == {"gaussian", "copula", "binning", "ksg"}
    for v in out.values():
        assert np.isfinite(v) and v >= 0.0


def test_estimators_agree_on_strong_signal():
    x, y = _bivariate_normal(0.9, 5000, seed=13)
    out = estimate_mi(x, y)
    assert min(out.values()) > 0.5


def test_short_series_returns_zero_not_error():
    assert gaussian_mi(np.array([1.0, 2.0]), np.array([1.0, 2.0])) >= 0.0
    assert ksg_mi(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), k=3) == 0.0
