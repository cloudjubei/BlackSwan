import numpy as np
import pytest

from trainer.trading_costs import (
    net_return_series,
    portfolio_gross_returns,
    turnover_series,
)


def test_turnover_first_period_is_entry_from_cash():
    w = np.array([[1.0, 0.0], [1.0, 0.0]])
    t = turnover_series(w)
    assert t[0] == pytest.approx(1.0)
    assert t[1] == pytest.approx(0.0)


def test_turnover_constant_weights_zero_after_entry():
    w = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    t = turnover_series(w)
    assert t[0] == pytest.approx(1.0)
    assert np.allclose(t[1:], 0.0)


def test_turnover_full_flip_is_two():
    w = np.array([[1.0], [-1.0]])
    t = turnover_series(w)
    assert t[1] == pytest.approx(2.0)


def test_turnover_partial_rebalance():
    w = np.array([[0.2, 0.8], [0.5, 0.5]])
    t = turnover_series(w)
    assert t[1] == pytest.approx(abs(0.5 - 0.2) + abs(0.5 - 0.8))


def test_gross_single_asset_equals_asset_return():
    w = np.array([[1.0], [1.0], [1.0]])
    r = np.array([[0.01], [-0.02], [0.03]])
    g = portfolio_gross_returns(w, r)
    assert np.allclose(g, r[:, 0])


def test_gross_is_contemporaneous_dot_product():
    w = np.array([[0.5, 0.5], [1.0, 0.0]])
    r = np.array([[0.10, -0.10], [0.04, 0.20]])
    g = portfolio_gross_returns(w, r)
    assert g[0] == pytest.approx(0.0)
    assert g[1] == pytest.approx(0.04)


def test_net_equals_gross_when_fee_zero():
    w = np.array([[1.0], [-1.0], [1.0]])
    r = np.array([[0.01], [0.02], [-0.01]])
    assert np.allclose(net_return_series(w, r, 0.0), portfolio_gross_returns(w, r))


def test_net_subtracts_fee_times_turnover():
    w = np.array([[1.0], [-1.0]])
    r = np.array([[0.0], [0.0]])
    fee = 0.001
    net = net_return_series(w, r, fee)
    assert net[0] == pytest.approx(-fee * 1.0)
    assert net[1] == pytest.approx(-fee * 2.0)


def test_net_zero_turnover_period_unaffected_by_fee():
    w = np.array([[1.0], [1.0]])
    r = np.array([[0.05], [0.07]])
    net = net_return_series(w, r, 0.01)
    assert net[1] == pytest.approx(0.07)


def test_higher_fee_never_raises_net():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((50, 3))
    r = rng.standard_normal((50, 3)) * 0.01
    lo = net_return_series(w, r, 0.0005)
    hi = net_return_series(w, r, 0.005)
    assert np.all(hi <= lo + 1e-12)


def test_shapes_must_match():
    with pytest.raises(ValueError):
        portfolio_gross_returns(np.zeros((3, 2)), np.zeros((3, 3)))


def test_empty_returns_empty():
    assert turnover_series(np.zeros((0, 2))).shape == (0,)
    assert net_return_series(np.zeros((0, 2)), np.zeros((0, 2)), 0.001).shape == (0,)
