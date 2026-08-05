"""Direct unit tests for the deterministic MaCrossoverStrategyModel baseline (variable-length MA crossover).

get_action compares a SHORT moving average to a LONG moving average of the trailing price window (via
env.get_price / env.current_step). Short above long by more than the band -> long (BUY=1); short below
long by more than the band -> flat (EXIT=2); inside the band or insufficient history -> HOLD=0 (keep the
current position, the classic BLL/Grobys band filter that suppresses whipsaws). short_window=1 makes the
short MA the current price (the 1/N "variable MA" of Grobys 2020 and Brock et al. 1992). The env's
position-gating turns the repeated 1/2 stream into "hold long" / "stay flat".
"""

import types

from src.model.ma_crossover_strategy_model import MaCrossoverStrategyModel


def _env(prices, current_step, allow_shorting=False):
    return types.SimpleNamespace(
        current_step=current_step,
        get_price=lambda step: float(prices[step]),
        env_config=types.SimpleNamespace(allow_shorting=allow_shorting),
    )


def _model(short_window=1, long_window=3, band=0.0):
    m = MaCrossoverStrategyModel.__new__(MaCrossoverStrategyModel)
    m.ma_config = types.SimpleNamespace(
        short_window=short_window, long_window=long_window, band=band
    )
    return m


# --- get_id ---------------------------------------------------------------

def test_get_id_formats_type_and_windows():
    cfg = types.SimpleNamespace(
        model_type="ma_crossover",
        model_ma_crossover=types.SimpleNamespace(short_window=1, long_window=50, band=0.01),
    )
    assert MaCrossoverStrategyModel.get_id(None, cfg) == "ma_crossover_1_50_0~01"


# --- insufficient history ---------------------------------------------------

def test_hold_when_fewer_bars_than_long_window():
    # current_step 1 -> only 2 bars available (0,1) < long_window 3 -> HOLD.
    env = _env([100.0, 101.0, 102.0], current_step=1)
    assert _model(short_window=1, long_window=3).get_action(env, None) == 0


def test_hold_when_long_window_nonpositive():
    env = _env([100.0, 110.0], current_step=1)
    assert _model(short_window=1, long_window=0).get_action(env, None) == 0


# --- long when short MA above long MA ---------------------------------------

def test_long_when_short_ma_above_long_ma():
    # Rising series: price now (short=1 -> 130) well above the 4-bar long MA -> BUY/long.
    env = _env([100, 110, 120, 130], current_step=3)
    assert _model(short_window=1, long_window=4).get_action(env, None) == 1


def test_flat_when_short_ma_below_long_ma():
    # Falling series: price now (100) below the 4-bar long MA (avg 122.5) -> EXIT/flat.
    env = _env([160, 140, 120, 100], current_step=3)
    assert _model(short_window=1, long_window=4).get_action(env, None) == 2


# --- the band suppresses small crossings (BLL/Grobys whipsaw filter) --------

def test_hold_inside_band():
    # short MA (price=101) vs long MA (mean 101,100,100,100 = 100.25): +0.75% gap, inside a 1% band -> HOLD.
    env = _env([100, 100, 100, 101], current_step=3)
    m = _model(short_window=1, long_window=4, band=0.01)
    # sanity: with NO band the same bar is a BUY (short above long), proving the band is what holds it.
    assert _model(short_window=1, long_window=4, band=0.0).get_action(env, None) == 1
    assert m.get_action(env, None) == 0


def test_buy_when_gap_exceeds_band():
    # short MA (130) exceeds long MA by far more than a 1% band -> BUY even with the band on.
    env = _env([100, 110, 120, 130], current_step=3)
    assert _model(short_window=1, long_window=4, band=0.01).get_action(env, None) == 1


# --- the SHORT window is load-bearing (not just current price) --------------

def test_short_window_averages_multiple_bars():
    # short_window=2 -> short MA = mean(last 2) ; long_window=4 -> long MA = mean(last 4).
    # prices: last2 = (10+30)/2 = 20 ; last4 = (40+20+10+30)/4 = 25 -> short(20) < long(25) -> flat.
    env = _env([40, 20, 10, 30], current_step=3)
    assert _model(short_window=2, long_window=4).get_action(env, None) == 2


# --- long/short: SHORT the downtrend when the env allows shorting -----------

def test_short_on_downtrend_when_allow_shorting():
    # Falling series -> short MA below long MA. In a long+SHORT env, be SHORT (3) instead of flat (2):
    # long-only trend can only sit out downtrends; long/short profits from them.
    env = _env([160, 140, 120, 100], current_step=3, allow_shorting=True)
    assert _model(short_window=1, long_window=4).get_action(env, None) == 3


def test_flat_on_downtrend_when_shorting_disabled():
    # Same downtrend, long-only env -> still FLAT (2), never short (default behaviour unchanged).
    env = _env([160, 140, 120, 100], current_step=3, allow_shorting=False)
    assert _model(short_window=1, long_window=4).get_action(env, None) == 2


def test_long_unaffected_by_allow_shorting():
    # An uptrend is LONG (1) regardless of the shorting flag — shorting only changes the downtrend branch.
    env = _env([100, 110, 120, 130], current_step=3, allow_shorting=True)
    assert _model(short_window=1, long_window=4).get_action(env, None) == 1


# --- guards -----------------------------------------------------------------

def test_hold_when_long_ma_nonpositive():
    # A zero/garbage window must not divide-by-zero or emit a spurious signal.
    env = _env([0.0, 0.0, 0.0, 0.0], current_step=3)
    assert _model(short_window=1, long_window=4).get_action(env, None) == 0
