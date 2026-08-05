"""Direct unit tests for the deterministic BreakoutStrategyModel baseline (trading-range breakout, BLL 1992).

get_action compares the current price to the local resistance (max) and support (min) of the PRIOR
`window` bars (excluding the current bar, via env.get_price / env.current_step). Price breaking above
resistance by more than the band -> long (BUY=1); breaking below support by more than the band -> flat
(EXIT=2); inside the range -> HOLD=0 (keep the current position). The env's position-gating turns the
repeated 1/2 stream into "hold long" / "stay flat".
"""

import types

from src.model.breakout_strategy_model import BreakoutStrategyModel


def _env(prices, current_step, allow_shorting=False):
    return types.SimpleNamespace(
        current_step=current_step,
        get_price=lambda step: float(prices[step]),
        env_config=types.SimpleNamespace(allow_shorting=allow_shorting),
    )


def _breakout(window=3, band=0.0):
    m = BreakoutStrategyModel.__new__(BreakoutStrategyModel)
    m.breakout_config = types.SimpleNamespace(window=window, band=band)
    return m


def test_short_on_break_below_support_when_allow_shorting():
    # prior window (100,90,95) min=90 ; now=80 < 90 -> break below support -> SHORT (3) in a long+short env.
    env = _env([100, 90, 95, 80], current_step=3, allow_shorting=True)
    assert _breakout(window=3).get_action(env, None) == 3


def test_flat_on_break_below_support_when_shorting_disabled():
    env = _env([100, 90, 95, 80], current_step=3, allow_shorting=False)
    assert _breakout(window=3).get_action(env, None) == 2


def _model(window=3, band=0.0):
    m = BreakoutStrategyModel.__new__(BreakoutStrategyModel)
    m.breakout_config = types.SimpleNamespace(window=window, band=band)
    return m


# --- get_id ---------------------------------------------------------------

def test_get_id_formats_type_window_band():
    cfg = types.SimpleNamespace(
        model_type="breakout",
        model_breakout=types.SimpleNamespace(window=50, band=0.01),
    )
    assert BreakoutStrategyModel.get_id(None, cfg) == "breakout_50_0~01"


# --- insufficient history ---------------------------------------------------

def test_hold_when_fewer_bars_than_window():
    # current_step 2 < window 3 -> not enough PRIOR bars -> HOLD.
    env = _env([100.0, 101.0, 102.0], current_step=2)
    assert _model(window=3).get_action(env, None) == 0


def test_hold_when_window_nonpositive():
    env = _env([100.0, 110.0], current_step=1)
    assert _model(window=0).get_action(env, None) == 0


# --- breakout above resistance ---------------------------------------------

def test_long_when_price_breaks_above_prior_max():
    # prior window (100,110,105) max=110 ; now=120 > 110 -> BUY/long.
    env = _env([100, 110, 105, 120], current_step=3)
    assert _model(window=3).get_action(env, None) == 1


def test_flat_when_price_breaks_below_prior_min():
    # prior window (100,90,95) min=90 ; now=80 < 90 -> EXIT/flat.
    env = _env([100, 90, 95, 80], current_step=3)
    assert _model(window=3).get_action(env, None) == 2


def test_hold_when_price_inside_prior_range():
    # prior window (100,120,90) -> range [90,120] ; now=105 is inside -> HOLD.
    env = _env([100, 120, 90, 105], current_step=3)
    assert _model(window=3).get_action(env, None) == 0


# --- the band widens the range (BLL filter) --------------------------------

def test_band_suppresses_a_marginal_break():
    # prior max=110 ; now=111 is a break with no band (BUY) but inside a 1% band (110*1.01=111.1) -> HOLD.
    env = _env([100, 110, 105, 111], current_step=3)
    assert _model(window=3, band=0.0).get_action(env, None) == 1
    assert _model(window=3, band=0.01).get_action(env, None) == 0


# --- the window excludes the current bar -----------------------------------

def test_window_excludes_current_bar():
    # If the current bar were included, its own value would be the max and never break out. window=2 over
    # the PRIOR two bars (100,110) max=110 ; now=130 -> BUY, proving the current bar is excluded.
    env = _env([90, 100, 110, 130], current_step=3)
    assert _model(window=2).get_action(env, None) == 1
