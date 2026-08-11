"""Direct unit tests for the time-series-momentum / trend-following core (the published-anomaly battery).

TSMOM is the most-defended free anomaly (Moskowitz-Ooi-Pedersen 2012; the whole managed-futures/CTA
industry), so the refutation only means anything if the backtest is airtight. The failure modes here are
exactly the ones the cross-sectional line has (a misaligned N-symbol join fabricates P&L; a symbol ranked
before it has listed is survivorship bias) PLUS two the sign-of-trend rule adds: a vol scaler that peeks at
the bar it sizes, and a "trend" sign read on the same bar it trades. Each is pinned before the probe is
allowed to emit a number, so a green gate is a real result rather than a leak.
"""

import numpy as np
import pandas as pd
import pytest

from trainer import tsmom


def _frame(dates, prices):
    return pd.DataFrame({"timestamp_close": pd.to_datetime(dates), "price": prices})


def _matrix(**symbols):
    return tsmom.align_prices({k: _frame(*v) for k, v in symbols.items()})


# --- the rule: sign of the trailing return ------------------------------------------------------------


def test_trend_longs_the_uptrend_and_shorts_the_downtrend():
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    m = _matrix(
        UP=(dates, [100.0, 105, 110, 115, 120]),
        DOWN=(dates, [100.0, 95, 90, 85, 80]),
    )
    w = tsmom.build_weights(m, lookback=2, rebalance_days=1, signal="trend", weight_scheme="equal")
    last = w.iloc[-1]
    assert last["UP"] > 0 and last["DOWN"] < 0
    assert abs(last.abs().sum() - 1.0) < 1e-12  # fully invested at unit gross


def test_trend_inverse_is_the_exact_negation_of_trend_in_long_short_mode():
    # The mirror control: pre-registering both directions is what stops post-hoc sign cherry-picking. It holds
    # exactly whenever every asset has a non-zero, unambiguous trend sign (no flats to break the symmetry).
    dates = pd.date_range("2022-01-01", periods=8, freq="D")
    rng = np.random.default_rng(1)
    m = _matrix(**{s: (dates, list(100 + np.cumsum(rng.normal(0.3, 1, 8)))) for s in ("A", "B", "C")})
    kwargs = dict(lookback=2, rebalance_days=1, weight_scheme="equal")
    fwd = tsmom.build_weights(m, signal="trend", **kwargs)
    inv = tsmom.build_weights(m, signal="trend_inverse", **kwargs)
    pd.testing.assert_frame_equal(inv, -fwd)


def test_long_only_holds_the_uptrend_and_goes_flat_on_the_downtrend():
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    m = _matrix(
        UP=(dates, [100.0, 105, 110, 115, 120]),
        DOWN=(dates, [100.0, 95, 90, 85, 80]),
    )
    w = tsmom.build_weights(
        m, lookback=2, rebalance_days=1, signal="trend", weight_scheme="equal", long_only=True
    )
    last = w.iloc[-1]
    assert last["UP"] > 0 and last["DOWN"] == 0.0
    assert abs(last.abs().sum() - 1.0) < 1e-12  # the single long carries the whole book


# --- vol-scaling: risk-weighting, and the leakage it can hide -----------------------------------------


def test_invvol_gives_the_calmer_asset_the_bigger_weight():
    # Two assets trending up at the same sign; the low-vol one must carry MORE risk budget than the choppy
    # one. Equal weighting would give them the same magnitude — this is what distinguishes the scheme.
    dates = pd.date_range("2022-01-01", periods=40, freq="D")
    calm = 100 * np.cumprod(1 + np.full(40, 0.002))
    rng = np.random.default_rng(2)
    choppy = 100 * np.cumprod(1 + 0.002 + rng.normal(0, 0.03, 40))
    m = _matrix(CALM=(dates, list(calm)), CHOPPY=(dates, list(choppy)))
    w = tsmom.build_weights(
        m, lookback=5, rebalance_days=1, signal="trend", weight_scheme="invvol", vol_span=10
    )
    last = w.iloc[-1]
    assert last["CALM"] > 0 and last["CHOPPY"] > 0
    assert abs(last["CALM"]) > abs(last["CHOPPY"])  # inverse-vol -> calmer asset, bigger bet
    assert abs(last.abs().sum() - 1.0) < 1e-12


def test_an_unknown_weight_scheme_is_refused_rather_than_silently_equal_weighted():
    dates = pd.date_range("2022-01-01", periods=4, freq="D")
    m = _matrix(A=(dates, [100.0, 101, 102, 103]), B=(dates, [100.0, 99, 98, 97]))
    with pytest.raises(ValueError):
        tsmom.build_weights(m, lookback=1, rebalance_days=1, signal="trend", weight_scheme="riskparity")


def test_a_non_positive_lookback_is_refused_rather_than_reading_the_future():
    # sign() of `s / s.shift(lookback)`: a negative shift ratios against a bar that has not happened, so the
    # trend sign becomes a peek at the future. Nothing downstream would notice — the curve just looks good.
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    m = _matrix(A=(dates, [100.0, 110, 120, 130, 140]), B=(dates, [100.0, 90, 80, 70, 60]))
    for bad in (-2, 0):
        with pytest.raises(ValueError):
            tsmom.build_weights(m, lookback=bad, rebalance_days=1, signal="trend", weight_scheme="equal")


def test_an_unrecognised_signal_is_rejected_rather_than_silently_trend():
    dates = pd.date_range("2022-01-01", periods=4, freq="D")
    m = _matrix(A=(dates, [100.0, 110, 120, 130]), B=(dates, [100.0, 90, 80, 70]))
    with pytest.raises(ValueError):
        tsmom.build_weights(m, lookback=1, rebalance_days=1, signal="meanrev", weight_scheme="equal")


# --- survivorship: a symbol invisible until it has enough history -------------------------------------


def test_a_symbol_is_not_traded_before_it_has_enough_history():
    dates = pd.date_range("2022-01-01", periods=8, freq="D")
    m = _matrix(
        A=(dates, [100.0, 101, 102, 103, 104, 105, 106, 107]),
        B=(dates, [100.0, 99, 98, 97, 96, 95, 94, 93]),
        LATE=(dates[6:], [10.0, 11.0]),
    )
    w = tsmom.build_weights(m, lookback=3, rebalance_days=1, signal="trend", weight_scheme="equal")
    assert (w["LATE"] == 0.0).all()  # LATE never has 3 prior bars inside this span


# --- causality: the whole reason the refutation is trustworthy ----------------------------------------
#
# Parametrized over BOTH the signal arm and the weight scheme, so a lookahead that only bites the inverse arm
# or only the vol scaler is still caught, and a third arm/scheme inherits both guards the day it is declared.


@pytest.mark.parametrize("signal", tsmom.SIGNALS)
@pytest.mark.parametrize("scheme", tsmom.WEIGHT_SCHEMES)
def test_weights_are_time_prefix_causal(signal, scheme):
    """Corrupt every bar from `cut` onward with a DIFFERENT per-symbol factor; the book held into `cut` and
    everything before it must be byte-identical, and something after it must move (or the test proves nothing).
    Row `cut` is included because it is the book decided at `cut-1` and must survive the corruption — the
    one-bar slack a lookahead bug actually lives in."""
    dates = pd.date_range("2022-01-01", periods=24, freq="D")
    rng = np.random.default_rng(0)
    base = {s: (dates, list(100 + np.cumsum(rng.normal(0, 1, 24)))) for s in ("A", "B", "C", "D")}
    m = _matrix(**base)
    kwargs = dict(lookback=3, rebalance_days=2, signal=signal, weight_scheme=scheme, vol_span=5)
    clean = tsmom.build_weights(m, **kwargs)
    for cut in range(8, 22):
        dirty_m = m.copy()
        dirty_m.iloc[cut:] = dirty_m.iloc[cut:] * [5.0, 0.2, 1.0, 3.0]
        dirty = tsmom.build_weights(dirty_m, **kwargs)
        pd.testing.assert_frame_equal(clean.iloc[: cut + 1], dirty.iloc[: cut + 1])
        assert not clean.iloc[cut + 1 :].equals(dirty.iloc[cut + 1 :])


@pytest.mark.parametrize("scheme", tsmom.WEIGHT_SCHEMES)
def test_a_trend_is_never_traded_on_its_own_bar(scheme):
    # Shock ONE bar hard enough to flip a sign, then read the book either side: the weight held INTO that bar
    # was decided a bar earlier and must not see it, while the NEXT row must move. The second half is what
    # stops this collapsing into "row zero is flat", which is true of any book, leak or not.
    dates = pd.date_range("2022-01-01", periods=10, freq="D")
    base = {
        "A": [100.0, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "B": [100.0, 99, 98, 97, 96, 95, 94, 93, 92, 91],
        "C": [100.0, 100.5, 101, 101.5, 102, 102.5, 103, 103.5, 104, 104.5],
    }
    shocked = {s: list(p) for s, p in base.items()}
    shocked["B"][5] = 400.0
    kwargs = dict(lookback=2, rebalance_days=1, signal="trend", weight_scheme=scheme, vol_span=4)
    clean = tsmom.build_weights(_matrix(**{s: (dates, p) for s, p in base.items()}), **kwargs)
    dirty = tsmom.build_weights(_matrix(**{s: (dates, p) for s, p in shocked.items()}), **kwargs)
    assert clean.iloc[0].abs().sum() == 0.0
    pd.testing.assert_frame_equal(clean.iloc[:6], dirty.iloc[:6])
    assert not clean.iloc[6].equals(dirty.iloc[6])


# --- the lever as a CELL sees it (the run() wiring the persisted records are read through) -------------

_RUN_CFG = {
    "universe": "diversified",
    "walk_forward_window": "2024",
    "lookback": 3,
    "rebalance_days": 1,
    "vol_span": 4,
    "transaction_fee": 0.0,
}


def _osc_frames():
    """Two assets in opposite phase across the train tail and the whole 2024 test window: trend and its
    inverse land far apart here, so a cell that silently ran the other arm cannot hide in a tolerance."""
    dates = pd.date_range("2023-06-01", periods=420, freq="D")
    steps = np.arange(420)
    return {
        "GOLD": _frame(dates, list(100.0 * (1.0 + 0.1 * np.sin(steps / 9.0)))),
        "SPY": _frame(dates, list(100.0 * (1.0 + 0.1 * np.cos(steps / 9.0)))),
    }


def _stub_loader(monkeypatch, frames):
    calls = []

    def _fake(symbols, pairs):
        calls.append((tuple(symbols), len(pairs)))
        return dict(frames)

    monkeypatch.setattr(tsmom, "_load_universe", _fake)
    return calls


def test_a_reversal_of_the_arm_actually_changes_the_cell(monkeypatch):
    _stub_loader(monkeypatch, _osc_frames())
    fwd = tsmom.run({**_RUN_CFG, "signal": "trend"})
    inv = tsmom.run({**_RUN_CFG, "signal": "trend_inverse"})
    assert abs(fwd["objective"] - inv["objective"]) > 1.0
    assert fwd["config"]["signal"] == "trend"  # the record must say which arm produced it


def test_a_typod_signal_kills_the_cell_before_any_data_is_read(monkeypatch):
    calls = _stub_loader(monkeypatch, _osc_frames())
    with pytest.raises(SystemExit) as raised:
        tsmom.run({**_RUN_CFG, "signal": "meanrev"})
    assert "meanrev" in str(raised.value)
    assert calls == []


def test_the_cell_reports_the_gate_metric_and_window(monkeypatch):
    _stub_loader(monkeypatch, _osc_frames())
    out = tsmom.run(dict(_RUN_CFG))
    assert "oos_sharpe" in out["metrics"]
    assert out["walk_forward_window"] == "2024"
    assert out["metrics"]["bars"] > 0
