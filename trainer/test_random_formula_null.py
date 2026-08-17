import numpy as np
import pytest

from trainer.random_formula_null import (
    formula_sharpe,
    null_pvalue,
    random_formula,
    random_formula_null,
    signal_to_weights,
)


def _panel(rng, t, n, phi=0.0):
    r = np.zeros((t, n))
    r[0] = rng.standard_normal(n) * 0.01
    for i in range(1, t):
        r[i] = phi * r[i - 1] + np.sqrt(max(1e-9, 1 - phi ** 2)) * rng.standard_normal(n) * 0.01
    return r


def test_weights_are_dollar_neutral_and_unit_gross():
    s = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    w = signal_to_weights(s)
    assert np.allclose(w.sum(axis=1), 0.0)
    assert np.allclose(np.abs(w).sum(axis=1), 1.0)


def test_weights_constant_row_is_zero():
    w = signal_to_weights(np.array([[5.0, 5.0, 5.0]]))
    assert np.allclose(w, 0.0)


def test_random_formula_deterministic_given_seed():
    feats = [_panel(np.random.default_rng(1), 40, 5)]
    a = random_formula(np.random.default_rng(7), feats, depth=3)
    b = random_formula(np.random.default_rng(7), feats, depth=3)
    assert np.allclose(a, b)
    assert a.shape == (40, 5)
    assert np.all(np.isfinite(a))


def test_random_formula_varies_with_seed():
    feats = [_panel(np.random.default_rng(1), 40, 5)]
    a = random_formula(np.random.default_rng(7), feats, depth=3)
    c = random_formula(np.random.default_rng(8), feats, depth=3)
    assert not np.allclose(a, c)


def test_sharpe_scale_invariant():
    rng = np.random.default_rng(3)
    r = _panel(rng, 200, 6, phi=0.5)
    s = np.vstack([np.zeros((1, 6)), r[:-1]])
    assert formula_sharpe(s, r) == pytest.approx(formula_sharpe(5.0 * s, r), rel=1e-9)


def test_sharpe_sign_flip_negates():
    rng = np.random.default_rng(4)
    r = _panel(rng, 200, 6, phi=0.5)
    s = np.vstack([np.zeros((1, 6)), r[:-1]])
    assert formula_sharpe(-s, r) == pytest.approx(-formula_sharpe(s, r), rel=1e-9)


def test_no_lookahead_contemporaneous_signal_earns_nothing_on_iid():
    rng = np.random.default_rng(5)
    r = _panel(rng, 600, 8, phi=0.0)
    sharpe = formula_sharpe(r, r)
    assert abs(sharpe) < 1.0


def test_lagged_predictor_earns_on_autocorrelated_panel():
    rng = np.random.default_rng(6)
    r = _panel(rng, 800, 8, phi=0.6)
    signal = np.vstack([np.zeros((1, 8)), r[:-1]])
    assert formula_sharpe(signal, r) > 1.0


def test_higher_fee_never_raises_null_sharpes():
    feats = [_panel(np.random.default_rng(2), 300, 6, phi=0.3)]
    r = _panel(np.random.default_rng(9), 300, 6, phi=0.3)
    lo = random_formula_null(np.random.default_rng(11), feats, r, depth=3, k=40, fee=0.0)
    hi = random_formula_null(np.random.default_rng(11), feats, r, depth=3, k=40, fee=0.003)
    assert np.all(hi <= lo + 1e-9)


def test_null_has_no_systematic_edge():
    feats = [_panel(np.random.default_rng(2), 500, 8, phi=0.2)]
    r = _panel(np.random.default_rng(21), 500, 8, phi=0.2)
    null = random_formula_null(np.random.default_rng(31), feats, r, depth=3, k=120, fee=0.0)
    assert null.shape == (120,)
    assert np.all(np.isfinite(null))
    assert abs(np.median(null)) < 0.5


def test_pvalue_monotone_and_bounded():
    null = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert null_pvalue(2.0, null) == pytest.approx(1 / 6)
    assert null_pvalue(-2.0, null) == pytest.approx(6 / 6)
    assert null_pvalue(0.0, null) <= null_pvalue(-0.5, null)
    p = null_pvalue(0.3, null)
    assert 0.0 < p <= 1.0


def test_pvalue_continuity_correction():
    null = np.zeros(9)
    assert null_pvalue(100.0, null) == pytest.approx(1 / 10)
