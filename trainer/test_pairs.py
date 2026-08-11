"""Direct unit tests for the pairs / statistical-arbitrage core (the published-anomaly battery).

Distance pairs trading (Gatev-Goetzmann-Rouwenhorst 2006): over a FORMATION window pick the closest-moving
pairs, then in the TRADING window fade divergences (short the rich leg / long the cheap leg) and close on
convergence. The failure modes pinned here: pair SELECTION and the spread's mean/std must come only from the
formation window (choosing pairs on the trading window is peeking), the entry/exit state machine must be
exactly the mean-reversion rule, and the book must be one bar behind the spread it reads.
"""

import numpy as np
import pandas as pd
import pytest

from trainer import pairs


def _frame(dates, prices):
    return pd.DataFrame({"timestamp_close": pd.to_datetime(dates), "price": prices})


def _matrix(**symbols):
    return pairs.align_prices({k: _frame(*v) for k, v in symbols.items()})


# --- the mean-reversion state machine -----------------------------------------------------------------


def test_position_enters_on_divergence_and_exits_on_convergence():
    # z path: calm, spikes ABOVE +entry (short the spread = -1), drifts back through 0 (exit), spikes BELOW
    # -entry (long the spread = +1), back through 0 (exit).
    z = [0.0, 0.5, 2.5, 1.5, -0.2, -0.5, -2.5, -1.0, 0.3]
    pos = pairs.pair_position_series(z, entry=2.0)
    assert pos == [0, 0, -1, -1, 0, 0, 1, 1, 0]


def test_position_holds_between_entry_and_the_zero_cross():
    # Once short at z>entry, stay short through every bar until z crosses 0 — not merely when it drops back
    # under `entry`. A rule that exits at `entry` instead of 0 would book a different (smaller) reversion.
    z = [0.0, 2.2, 1.9, 1.1, 0.4, -0.1, 0.2]
    pos = pairs.pair_position_series(z, entry=2.0)
    assert pos == [0, -1, -1, -1, -1, 0, 0]


def test_select_pairs_picks_the_closest_moving_pair_not_the_diverging_one():
    # A and B move almost identically; C wanders off. The closest pair by sum-of-squared-distance is (A, B).
    dates = pd.date_range("2022-01-01", periods=30, freq="D")
    rng = np.random.default_rng(0)
    common = np.cumsum(rng.normal(0, 0.01, 30))
    a = 100 * np.exp(common + rng.normal(0, 0.0005, 30))
    b = 100 * np.exp(common + rng.normal(0, 0.0005, 30))
    c = 100 * np.exp(np.cumsum(rng.normal(0.01, 0.02, 30)))
    m = _matrix(A=(dates, list(a)), B=(dates, list(b)), C=(dates, list(c)))
    norm = pairs.normalize(m)
    chosen = pairs.select_pairs(norm, k=1)
    assert chosen == [("A", "B")]


# --- leakage: selection + spread params are PAST-only, book is one bar behind ------------------------


def test_pair_selection_uses_only_the_formation_window(monkeypatch):
    # Two candidate pairs; which is "closest" flips between the formation window and the trading window. The
    # book must be built on the FORMATION ranking — a selection that peeked at the trading window would hold
    # the other pair. Verified by checking the chosen pair via the formation slice directly.
    dates = pd.date_range("2022-01-01", periods=40, freq="D")
    rng = np.random.default_rng(1)
    f = np.cumsum(rng.normal(0, 0.01, 40))
    # In formation (first 20) A~B; in trading (last 20) A~C. Formation must pick (A,B).
    a = 100 * np.exp(f)
    b = 100 * np.exp(f + np.concatenate([rng.normal(0, 0.0003, 20), rng.normal(0, 0.05, 20)]))
    c = 100 * np.exp(f + np.concatenate([rng.normal(0, 0.05, 20), rng.normal(0, 0.0003, 20)]))
    m = _matrix(A=(dates, list(a)), B=(dates, list(b)), C=(dates, list(c)))
    form = m.iloc[:20]
    chosen = pairs.select_pairs(pairs.normalize(form), k=1)
    assert chosen == [("A", "B")]  # closest in FORMATION, though A~C in the trading window


def test_weights_are_time_prefix_causal():
    # Corrupt every price from `cut` onward; the book held into `cut` and everything before must be identical,
    # and something after must move. Selection is fixed on the formation window (pre-test), so only the z-path
    # inside the trading window can change the book — one bar behind the spread.
    dates = pd.date_range("2022-01-01", periods=80, freq="D")
    rng = np.random.default_rng(2)
    common = np.cumsum(rng.normal(0, 0.01, 80))
    frames = {
        "A": (dates, list(100 * np.exp(common + rng.normal(0, 0.002, 80)))),
        "B": (dates, list(100 * np.exp(common + rng.normal(0, 0.002, 80)))),
        "D": (dates, list(100 * np.exp(common + rng.normal(0, 0.002, 80)))),
    }
    m = _matrix(**frames)
    test_start = dates[40]
    kwargs = dict(formation_days=40, k=1, entry=1.5, signal="meanrev", test_start=test_start)
    clean = pairs.build_weights(m, **kwargs)
    for cut in range(45, 78):
        dirty = m.copy()
        dirty.iloc[cut:] = dirty.iloc[cut:] * [1.5, 0.7, 1.2]
        w = pairs.build_weights(dirty, **kwargs)
        pd.testing.assert_frame_equal(clean.iloc[: cut + 1], w.iloc[: cut + 1])
        assert not clean.iloc[cut + 1 :].equals(w.iloc[cut + 1 :])


def test_meanrev_inverse_is_the_exact_negation():
    dates = pd.date_range("2022-01-01", periods=60, freq="D")
    rng = np.random.default_rng(3)
    common = np.cumsum(rng.normal(0, 0.01, 60))
    m = _matrix(
        A=(dates, list(100 * np.exp(common + rng.normal(0, 0.003, 60)))),
        B=(dates, list(100 * np.exp(common + rng.normal(0, 0.003, 60)))),
    )
    ts = dates[30]
    kwargs = dict(formation_days=30, k=1, entry=1.5, test_start=ts)
    fwd = pairs.build_weights(m, signal="meanrev", **kwargs)
    inv = pairs.build_weights(m, signal="meanrev_inverse", **kwargs)
    pd.testing.assert_frame_equal(inv, -fwd)


def test_each_pair_leg_is_dollar_neutral():
    dates = pd.date_range("2022-01-01", periods=60, freq="D")
    rng = np.random.default_rng(4)
    common = np.cumsum(rng.normal(0, 0.01, 60))
    m = _matrix(
        A=(dates, list(100 * np.exp(common + rng.normal(0, 0.02, 60)))),  # noisier -> real divergences to trade
        B=(dates, list(100 * np.exp(common + rng.normal(0, 0.02, 60)))),
    )
    w = pairs.build_weights(m, formation_days=30, k=1, entry=1.0, signal="meanrev", test_start=dates[30])
    assert w.abs().sum().sum() > 0  # it actually trades
    assert (w.sum(axis=1).abs() < 1e-9).all()  # every bar dollar-neutral (long one leg, short the other)


def test_an_unknown_signal_is_refused():
    dates = pd.date_range("2022-01-01", periods=20, freq="D")
    m = _matrix(A=(dates, list(100 + np.arange(20.0))), B=(dates, list(100 + np.arange(20.0))))
    with pytest.raises(ValueError):
        pairs.build_weights(m, formation_days=10, k=1, entry=1.5, signal="trend", test_start=dates[10])


# --- the cell wiring ----------------------------------------------------------------------------------

_RUN_CFG = {
    "universe": "diversified",
    "walk_forward_window": "2024",
    "formation_days": 252,
    "k": 5,
    "entry": 2.0,
    "transaction_fee": 0.0,
}


def _cointegrated_frames():
    """A basket where several pairs co-move (a shared factor) plus idiosyncratic noise, spanning the 2023 train
    tail + the 2024 test window, so pairs form on 2023 and trade in 2024."""
    dates = pd.date_range("2023-01-01", periods=560, freq="D")
    rng = np.random.default_rng(9)
    common = np.cumsum(rng.normal(0, 0.01, 560))
    out = {}
    for s in ["GOLD", "SILVER", "COPPER", "WTI", "CORN", "WHEAT", "SPY", "TLT", "IEF", "UUP"]:
        out[s] = _frame(dates, list(100 * np.exp(common + np.cumsum(rng.normal(0, 0.006, 560)))))
    return out


def _stub_loader(monkeypatch, frames):
    calls = []

    def _fake(symbols, pairs_):
        calls.append((tuple(symbols), len(pairs_)))
        return dict(frames)

    monkeypatch.setattr(pairs, "_load_universe", _fake)
    return calls


def test_a_reversal_of_the_arm_changes_the_cell(monkeypatch):
    _stub_loader(monkeypatch, _cointegrated_frames())
    fwd = pairs.run({**_RUN_CFG, "signal": "meanrev"})
    inv = pairs.run({**_RUN_CFG, "signal": "meanrev_inverse"})
    assert abs(fwd["objective"] - inv["objective"]) > 0.1
    assert fwd["config"]["signal"] == "meanrev"


def test_a_typod_signal_kills_the_cell_before_any_data_is_read(monkeypatch):
    calls = _stub_loader(monkeypatch, _cointegrated_frames())
    with pytest.raises(SystemExit) as raised:
        pairs.run({**_RUN_CFG, "signal": "trend"})
    assert "trend" in str(raised.value)
    assert calls == []


def test_the_cell_reports_the_gate_metric_and_window(monkeypatch):
    _stub_loader(monkeypatch, _cointegrated_frames())
    out = pairs.run(dict(_RUN_CFG))
    assert "oos_sharpe" in out["metrics"]
    assert out["walk_forward_window"] == "2024"
    assert out["metrics"]["bars"] > 0
