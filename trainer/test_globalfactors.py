"""Direct unit tests for the Baltussen (2021) global-factor replication core.

This is the paper's FOIL: reproduce the style-factor premiums Baltussen-Swinkels-van Vliet report as robust,
then subject them to our discipline. The module is deliberately thin — it REUSES the already-mutation-proven
build_weights of tsmom (trend), xsection (momentum, and value = long-run reversal) and lowvol (low-beta), and
the only genuinely new object is the equal-risk DIVERSIFIED combination. So the tests here pin the two things
those component tests do not already cover: (1) value is long-run reversal at the value horizon (distinct from
12-1 momentum), and (2) the diversified book is the unit-gross average of its components AND stays one bar
behind — a linear combination of causal books must not manufacture look-ahead.
"""

import numpy as np
import pandas as pd
import pytest

from trainer import globalfactors as gf


def _frame(dates, prices):
    return pd.DataFrame({"timestamp_close": pd.to_datetime(dates), "price": prices})


def _matrix(**symbols):
    return gf.align_prices({k: _frame(*v) for k, v in symbols.items()})


def _panel(n=1400, seed=0):
    dates = pd.bdate_range("2006-01-02", periods=n)
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 0.01, n))
    frames = {}
    for i, s in enumerate(["GOLD", "SILVER", "COPPER", "WTI", "CORN", "WHEAT", "SPY", "TLT"]):
        drift = 0.0003 * (1 if i % 2 else -1)
        frames[s] = _frame(dates, list(100 * np.exp(common * (0.5 + 0.1 * i) + np.cumsum(rng.normal(drift, 0.012, n)))))
    return gf.align_prices(frames)


# --- value is long-run reversal, distinct from momentum ------------------------------------------------


def test_value_is_the_long_run_reversal_of_the_value_horizon_not_momentum():
    # A basket where the 5y (value) ranking and the 1y (momentum) ranking DISAGREE: an asset that fell over 5y
    # but rose in the last year is a value LONG (5y loser) yet a momentum LONG (1y winner) — so the value book
    # must equal xsection reversal at the value horizon, not the momentum book.
    dates = pd.bdate_range("2006-01-02", periods=1400)
    n = len(dates)
    t = np.arange(n)
    # LONGVAL: big 5y decline then a recent rally; WINNER: steady 5y riser; FLAT: sideways
    longval = 200 * np.exp(-0.0004 * t) * (1 + 0.3 * (t > n - 260) * (t - (n - 260)) / 260).clip(1)
    winner = 100 * np.exp(0.0004 * t)
    flat = 100 * np.ones(n) + np.sin(t / 50)
    m = _matrix(LONGVAL=(dates, list(longval)), WINNER=(dates, list(winner)), FLAT=(dates, list(flat)))
    start = dates[1300]
    value = gf.factor_weights(m, "value", start=start, value_lookback=1000, k=1, rebalance_days=5)
    from trainer import xsection
    expect = xsection.build_weights(m, 1000, 1, 5, False, "reversal", start=start)
    pd.testing.assert_frame_equal(value, expect)


def test_the_five_factor_names_and_diversified_are_the_public_surface():
    assert set(gf.FACTORS) == {"trend", "momentum", "value", "lowbeta", "diversified"}
    assert gf.SIGNALS == ("published", "inverse")


def test_an_unknown_factor_or_signal_is_refused():
    m = _panel(400)
    with pytest.raises(ValueError):
        gf.factor_weights(m, "carry", start=m.index[300])
    with pytest.raises(ValueError):
        gf.build_weights(m, "trend", signal="fade", start=m.index[300])


# --- the diversified combination -----------------------------------------------------------------------


def test_diversified_is_the_unit_gross_average_of_its_components():
    m = _panel(1400)
    start = m.index[1300]
    kw = dict(lookback=126, value_lookback=800, span=126, k=2, rebalance_days=10, vol_span=40)
    div = gf.factor_weights(m, "diversified", start=start, **kw)
    comps = [gf._unit_gross(gf.factor_weights(m, f, start=start, **kw)) for f in gf.DIVERSIFIED_LEGS]
    expect = gf._unit_gross(sum(comps) / len(comps))
    pd.testing.assert_frame_equal(div, expect)
    # every traded bar is at unit gross (or flat before warm-up)
    g = div.abs().sum(axis=1)
    assert ((((g - 1.0).abs()) < 1e-9) | (g < 1e-12)).all()


def test_inverse_is_the_exact_negation_for_every_factor():
    m = _panel(1400)
    start = m.index[1300]
    for f in gf.FACTORS:
        fwd = gf.build_weights(m, f, signal="published", start=start, lookback=126, value_lookback=800, span=126, k=2, rebalance_days=10)
        inv = gf.build_weights(m, f, signal="inverse", start=start, lookback=126, value_lookback=800, span=126, k=2, rebalance_days=10)
        pd.testing.assert_frame_equal(inv, -fwd)


def test_diversified_is_time_prefix_causal():
    # A linear combination of causal books must itself be causal: corrupt every price from `cut`, and the
    # combined book held into `cut` must be byte-identical, with something after it moving.
    m = _panel(900, seed=3)
    start = m.index[500]
    kw = dict(lookback=60, value_lookback=300, span=60, k=2, rebalance_days=5, vol_span=20)
    clean = gf.factor_weights(m, "diversified", start=start, **kw)
    cols = list(m.columns)
    for cut in range(560, 890, 40):
        dirty = m.copy()
        dirty.iloc[cut:] = dirty.iloc[cut:] * (1.0 + 0.3 * np.arange(1, len(cols) + 1))
        w = gf.factor_weights(dirty, "diversified", start=start, **kw)
        pd.testing.assert_frame_equal(clean.iloc[: cut + 1], w.iloc[: cut + 1])
        assert not clean.iloc[cut + 1 :].equals(w.iloc[cut + 1 :])


# --- the cell wiring -----------------------------------------------------------------------------------

_RUN_CFG = {"universe": "diversified", "walk_forward_window": "2020", "factor": "diversified", "transaction_fee": 0.0}


def _stub_loader(monkeypatch, frames):
    calls = []

    def _fake(symbols, pairs):
        calls.append((tuple(symbols), len(pairs)))
        return {s: frames[s] for s in symbols if s in frames}

    monkeypatch.setattr(gf, "_load_universe", _fake)
    return calls


def _loader_frames():
    dates = pd.bdate_range("2006-01-02", periods=3800)
    rng = np.random.default_rng(7)
    common = np.cumsum(rng.normal(0, 0.01, len(dates)))
    out = {}
    for i, s in enumerate(gf.UNIVERSES["diversified"]):
        out[s] = _frame(dates, list(100 * np.exp(common * (0.4 + 0.08 * i) + np.cumsum(rng.normal(0.0002 * (-1) ** i, 0.012, len(dates))))))
    return out


def test_a_reversal_of_the_arm_changes_the_cell(monkeypatch):
    _stub_loader(monkeypatch, _loader_frames())
    fwd = gf.run({**_RUN_CFG, "signal": "published"})
    inv = gf.run({**_RUN_CFG, "signal": "inverse"})
    assert abs(fwd["objective"] - inv["objective"]) > 0.1
    assert fwd["config"]["factor"] == "diversified"


def test_a_typod_factor_kills_the_cell_before_any_data_is_read(monkeypatch):
    calls = _stub_loader(monkeypatch, _loader_frames())
    with pytest.raises(SystemExit) as raised:
        gf.run({**_RUN_CFG, "factor": "carry"})
    assert "carry" in str(raised.value)
    assert calls == []


def test_the_cell_reports_the_gate_metric_and_window(monkeypatch):
    _stub_loader(monkeypatch, _loader_frames())
    out = gf.run(dict(_RUN_CFG))
    assert "oos_sharpe" in out["metrics"]
    assert out["walk_forward_window"] == "2020"
    assert out["metrics"]["bars"] > 0
