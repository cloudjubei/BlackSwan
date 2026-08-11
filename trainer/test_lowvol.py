"""Direct unit tests for the low-volatility / Betting-Against-Beta core (the published-anomaly battery).

The low-risk anomaly (Baker-Haugen; Frazzini-Pedersen 2014 BAB) claims low-risk assets earn higher
risk-adjusted returns, so a book LONG the low-risk / SHORT the high-risk names wins. The refutation is only
trustworthy if the ranking is airtight, so this pins the two leakage surfaces the rule adds: the risk score
(trailing vol OR trailing beta vs the basket) must be PAST-only, and it must be read a bar before it trades.
The survivorship + alignment guards are inherited from the cross-sectional line and re-checked here.
"""

import numpy as np
import pandas as pd
import pytest

from trainer import lowvol


def _frame(dates, prices):
    return pd.DataFrame({"timestamp_close": pd.to_datetime(dates), "price": prices})


def _matrix(**symbols):
    return lowvol.align_prices({k: _frame(*v) for k, v in symbols.items()})


def _trending(seed, drift, vol, n=60):
    rng = np.random.default_rng(seed)
    return list(100 * np.cumprod(1 + drift + rng.normal(0, vol, n)))


# --- the rule: long low-risk, short high-risk ---------------------------------------------------------


def test_lowrisk_longs_the_calm_asset_and_shorts_the_wild_one_by_vol():
    dates = pd.date_range("2022-01-01", periods=60, freq="D")
    m = _matrix(
        CALM=(dates, _trending(1, 0.001, 0.004)),
        MID=(dates, _trending(2, 0.001, 0.02)),
        WILD=(dates, _trending(3, 0.001, 0.06)),
    )
    w = lowvol.build_weights(m, span=20, rebalance_days=1, signal="lowrisk", rank_by="vol", k=1)
    last = w.iloc[-1]
    assert last["CALM"] > 0 and last["WILD"] < 0 and last["MID"] == 0.0
    assert abs(last.sum()) < 1e-12  # dollar-neutral
    assert abs(last.abs().sum() - 2.0) < 1e-12  # gross 2 (one long, one short)


def test_lowrisk_inverse_is_the_exact_negation_in_long_short_mode():
    dates = pd.date_range("2022-01-01", periods=60, freq="D")
    m = _matrix(
        A=(dates, _trending(1, 0.001, 0.006)),
        B=(dates, _trending(2, 0.001, 0.02)),
        C=(dates, _trending(3, 0.001, 0.05)),
        D=(dates, _trending(4, 0.001, 0.035)),
    )
    kwargs = dict(span=20, rebalance_days=1, rank_by="vol", k=2)
    fwd = lowvol.build_weights(m, signal="lowrisk", **kwargs)
    inv = lowvol.build_weights(m, signal="lowrisk_inverse", **kwargs)
    pd.testing.assert_frame_equal(inv, -fwd)


def test_ranking_by_beta_longs_the_low_beta_asset_and_shorts_the_high_beta_one():
    # LOW is nearly flat (low beta to the equal-weight basket), HIGH swings ~2x a shared factor. Ranking by
    # beta must long LOW and short HIGH, distinct from a pure vol rank.
    dates = pd.date_range("2022-01-01", periods=80, freq="D")
    rng = np.random.default_rng(7)
    factor = rng.normal(0, 0.02, 80)
    lowb = list(100 * np.cumprod(1 + 0.05 * factor + rng.normal(0, 0.001, 80)))
    midb = list(100 * np.cumprod(1 + 1.0 * factor + rng.normal(0, 0.001, 80)))
    highb = list(100 * np.cumprod(1 + 2.0 * factor + rng.normal(0, 0.001, 80)))
    m = _matrix(LOW=(dates, lowb), MID=(dates, midb), HIGH=(dates, highb))
    w = lowvol.build_weights(m, span=30, rebalance_days=1, signal="lowrisk", rank_by="beta", k=1)
    last = w.iloc[-1]
    assert last["LOW"] > 0 and last["HIGH"] < 0


def test_an_unknown_signal_or_rank_key_is_refused_rather_than_silently_defaulting():
    dates = pd.date_range("2022-01-01", periods=6, freq="D")
    m = _matrix(A=(dates, [100.0, 101, 102, 103, 104, 105]), B=(dates, [100.0, 99, 98, 97, 96, 95]))
    with pytest.raises(ValueError):
        lowvol.build_weights(m, span=2, rebalance_days=1, signal="highrisk", rank_by="vol", k=1)
    with pytest.raises(ValueError):
        lowvol.build_weights(m, span=2, rebalance_days=1, signal="lowrisk", rank_by="skew", k=1)


def test_a_symbol_is_not_traded_before_it_has_enough_history():
    dates = pd.date_range("2022-01-01", periods=40, freq="D")
    m = _matrix(
        A=(dates, _trending(1, 0.001, 0.01, 40)),
        B=(dates, _trending(2, 0.001, 0.03, 40)),
        LATE=(dates[36:], [10.0, 11.0, 10.5, 11.2]),
    )
    w = lowvol.build_weights(m, span=20, rebalance_days=1, signal="lowrisk", rank_by="vol", k=1)
    assert (w["LATE"] == 0.0).all()


# --- causality: parametrized over BOTH rank keys and BOTH arms ----------------------------------------


@pytest.mark.parametrize("signal", lowvol.SIGNALS)
@pytest.mark.parametrize("rank_by", lowvol.RANK_KEYS)
def test_weights_are_time_prefix_causal(signal, rank_by):
    dates = pd.date_range("2022-01-01", periods=50, freq="D")
    rng = np.random.default_rng(0)
    base = {s: (dates, list(100 + np.cumsum(rng.normal(0, 1, 50)))) for s in ("A", "B", "C", "D")}
    m = _matrix(**base)
    kwargs = dict(span=8, rebalance_days=2, signal=signal, rank_by=rank_by, k=1)
    clean = lowvol.build_weights(m, **kwargs)
    for cut in range(16, 46):
        dirty_m = m.copy()
        dirty_m.iloc[cut:] = dirty_m.iloc[cut:] * [5.0, 0.2, 1.0, 3.0]
        dirty = lowvol.build_weights(dirty_m, **kwargs)
        pd.testing.assert_frame_equal(clean.iloc[: cut + 1], dirty.iloc[: cut + 1])
        assert not clean.iloc[cut + 1 :].equals(dirty.iloc[cut + 1 :])


@pytest.mark.parametrize("rank_by", lowvol.RANK_KEYS)
def test_the_risk_score_is_never_read_on_its_own_bar(rank_by):
    dates = pd.date_range("2022-01-01", periods=30, freq="D")
    rng = np.random.default_rng(3)
    base = {s: (dates, list(100 + np.cumsum(rng.normal(0, s_i + 0.5, 30)))) for s_i, s in enumerate(("A", "B", "C"))}
    shocked = {s: list(v[1]) for s, v in base.items()}
    shocked["B"][15] = shocked["B"][15] * 3.0  # a huge one-bar move spikes B's vol/beta at bar 15
    kwargs = dict(span=6, rebalance_days=1, signal="lowrisk", rank_by=rank_by, k=1)
    clean = lowvol.build_weights(_matrix(**{s: (dates, v[1]) for s, v in base.items()}), **kwargs)
    dirty = lowvol.build_weights(_matrix(**{s: (dates, shocked[s]) for s in base}), **kwargs)
    pd.testing.assert_frame_equal(clean.iloc[:16], dirty.iloc[:16])
    assert not clean.iloc[16:].equals(dirty.iloc[16:])


# --- the lever as a CELL sees it ----------------------------------------------------------------------

_RUN_CFG = {
    "universe": "diversified",
    "walk_forward_window": "2024",
    "span": 20,
    "rebalance_days": 5,
    "rank_by": "vol",
    "k": 3,
    "transaction_fee": 0.0,
}


def _osc_frames():
    dates = pd.date_range("2023-06-01", periods=420, freq="D")
    steps = np.arange(420)
    rng = np.random.default_rng(11)
    out = {}
    for i, s in enumerate(["GOLD", "SPY", "TLT", "WTI", "CORN", "IEF"]):
        vol = 0.004 + 0.01 * (i % 3)
        out[s] = _frame(dates, list(100 * np.cumprod(1 + 0.0003 + vol * np.sin(steps / (5 + i) + i) + rng.normal(0, 0.001, 420))))
    return out


def _stub_loader(monkeypatch, frames):
    calls = []

    def _fake(symbols, pairs):
        calls.append((tuple(symbols), len(pairs)))
        return dict(frames)

    monkeypatch.setattr(lowvol, "_load_universe", _fake)
    return calls


def test_a_reversal_of_the_arm_actually_changes_the_cell(monkeypatch):
    _stub_loader(monkeypatch, _osc_frames())
    fwd = lowvol.run({**_RUN_CFG, "signal": "lowrisk"})
    inv = lowvol.run({**_RUN_CFG, "signal": "lowrisk_inverse"})
    assert abs(fwd["objective"] - inv["objective"]) > 0.5
    assert fwd["config"]["signal"] == "lowrisk"


def test_a_typod_signal_kills_the_cell_before_any_data_is_read(monkeypatch):
    calls = _stub_loader(monkeypatch, _osc_frames())
    with pytest.raises(SystemExit) as raised:
        lowvol.run({**_RUN_CFG, "signal": "highrisk"})
    assert "highrisk" in str(raised.value)
    assert calls == []


def test_the_cell_reports_the_gate_metric_and_window(monkeypatch):
    _stub_loader(monkeypatch, _osc_frames())
    out = lowvol.run(dict(_RUN_CFG))
    assert "oos_sharpe" in out["metrics"]
    assert out["walk_forward_window"] == "2024"
    assert out["metrics"]["bars"] > 0
