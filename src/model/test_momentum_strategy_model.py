"""Direct unit tests for the deterministic MomentumStrategyModel baseline (time-series momentum).

get_action compares the current price to the price `lookback_periods` bars ago (via env.get_price /
env.current_step, so the lookback can exceed the observation window — Moskowitz-style long lookbacks).
Positive trailing return -> be long (BUY=1); non-positive -> be flat (EXIT=2); insufficient history or a
non-positive reference price -> HOLD=0. The env's position-gating turns a repeated BUY/EXIT into
"hold long" / "stay flat", so the 1/2 stream expresses the desired long/flat position each step.
"""

import types

from src.model.momentum_strategy_model import MomentumStrategyModel


def _env(prices, current_step, allow_shorting=False):
    return types.SimpleNamespace(
        current_step=current_step,
        get_price=lambda step: float(prices[step]),
        env_config=types.SimpleNamespace(allow_shorting=allow_shorting),
    )


def test_short_on_negative_momentum_when_allow_shorting():
    # now=90 vs 100 -> negative momentum. In a long+SHORT env, SHORT (3) instead of flat (2).
    env = _env([100.0, 90.0], current_step=1, allow_shorting=True)
    m = MomentumStrategyModel.__new__(MomentumStrategyModel)
    m.momentum_config = types.SimpleNamespace(lookback_periods=1)
    assert m.get_action(env, None) == 3


def test_flat_on_negative_momentum_when_shorting_disabled():
    env = _env([100.0, 90.0], current_step=1, allow_shorting=False)
    m = MomentumStrategyModel.__new__(MomentumStrategyModel)
    m.momentum_config = types.SimpleNamespace(lookback_periods=1)
    assert m.get_action(env, None) == 2


def _model(lookback_periods=3):
    m = MomentumStrategyModel.__new__(MomentumStrategyModel)
    m.momentum_config = types.SimpleNamespace(lookback_periods=lookback_periods)
    return m


# --- get_id ---------------------------------------------------------------

def test_get_id_formats_type_and_lookback():
    cfg = types.SimpleNamespace(
        model_type="momentum",
        model_momentum=types.SimpleNamespace(lookback_periods=252),
    )
    assert MomentumStrategyModel.get_id(None, cfg) == "momentum_252"


# --- insufficient history ---------------------------------------------------

def test_hold_when_fewer_bars_than_lookback():
    # current_step 2 < lookback 3 -> not enough history -> HOLD.
    env = _env([100.0, 101.0, 102.0], current_step=2)
    assert _model(lookback_periods=3).get_action(env, None) == 0


def test_hold_when_lookback_nonpositive():
    env = _env([100.0, 110.0], current_step=1)
    assert _model(lookback_periods=0).get_action(env, None) == 0


# --- long on positive momentum ---------------------------------------------

def test_long_when_trailing_return_positive():
    # now=prices[5]=110 vs prices[0]=100 over a 5-bar window -> +10% -> BUY/long.
    env = _env([100, 102, 104, 106, 108, 110], current_step=5)
    assert _model(lookback_periods=5).get_action(env, None) == 1


# --- flat on non-positive momentum -----------------------------------------

def test_flat_when_trailing_return_negative():
    # now=90 vs 100 -> -10% -> EXIT/flat.
    env = _env([100, 98, 96, 94, 92, 90], current_step=5)
    assert _model(lookback_periods=5).get_action(env, None) == 2


def test_flat_when_trailing_return_exactly_zero():
    # now == reference -> no positive momentum -> EXIT/flat (conservative).
    env = _env([100, 120, 80, 100], current_step=3)
    assert _model(lookback_periods=3).get_action(env, None) == 2


# --- the lookback window is load-bearing -----------------------------------

def test_lookback_selects_the_reference_bar():
    # Same series, two lookbacks -> opposite signals, proving the reference bar is step - lookback.
    prices = [100, 200, 150]  # step2=150
    assert _model(lookback_periods=2).get_action(_env(prices, 2), None) == 1  # 150 vs 100 -> up
    assert _model(lookback_periods=1).get_action(_env(prices, 2), None) == 2  # 150 vs 200 -> down


# --- guards -----------------------------------------------------------------

def test_hold_when_reference_price_nonpositive():
    # A zero/garbage reference price must not divide-by-zero or emit a spurious signal.
    env = _env([0.0, 10.0, 20.0, 30.0], current_step=3)
    assert _model(lookback_periods=3).get_action(env, None) == 0
