"""Direct unit tests for the deterministic TimeStrategyModel baseline.

get_action reads the most-recent bar's ``timestamp_close``, converts it to a UTC wall-clock, and
buys/sells when that clock (or its +1s offset) matches the configured ``time_buy`` / ``time_sell``
(stored as HHMM, scaled to HHMM00). We craft exact UTC epoch seconds and assert the emitted action.
"""

import calendar
import types
from datetime import datetime

import pandas as pd

from src.model.time_strategy_model import TimeStrategyModel


COLUMNS = ["timestamp_close", "price"]


def _epoch(hour, minute, second, day=1):
    # Build a UTC epoch matching datetime.utcfromtimestamp's interpretation (timegm = UTC tuple).
    return calendar.timegm(datetime(2021, 1, day, hour, minute, second).utctimetuple())


def _env(ts_close, *, lookback=1):
    item_size = len(COLUMNS)
    last_bar = [float(ts_close), 0.0]
    obs = [0.0] * (item_size * (lookback - 1)) + last_bar + [99999.0]
    env = types.SimpleNamespace(
        env_config=types.SimpleNamespace(lookback_window_size=lookback),
        df=types.SimpleNamespace(columns=pd.Index(COLUMNS)),
    )
    return env, obs


def _model(time_buy=1200, time_sell=1400):
    m = TimeStrategyModel.__new__(TimeStrategyModel)
    m.time_config = types.SimpleNamespace(time_buy=time_buy, time_sell=time_sell)
    return m


# --- get_id ---------------------------------------------------------------

def test_get_id_formats_type_and_times():
    cfg = types.SimpleNamespace(
        model_type="time",
        model_time=types.SimpleNamespace(time_buy=1200, time_sell=1400),
    )
    assert TimeStrategyModel.get_id(None, cfg) == "time_1200_1400"


# --- buy --------------------------------------------------------------------

def test_buy_when_close_exactly_on_buy_minute():
    # 12:00:00 close -> time_close 120000 == buy_time (1200*100) -> BUY.
    env, obs = _env(_epoch(12, 0, 0))
    assert _model(time_buy=1200).get_action(env, obs) == 1


def test_buy_via_one_second_offset():
    # 11:59:59 close + 1s -> 12:00:00 offset == buy_time -> BUY (handles :59 exchange close stamps).
    env, obs = _env(_epoch(11, 59, 59))
    assert _model(time_buy=1200).get_action(env, obs) == 1


def test_no_buy_when_off_by_two_seconds():
    # 11:59:58 close: neither it (115958) nor its +1s offset (115959) equals 120000.
    env, obs = _env(_epoch(11, 59, 58))
    assert _model(time_buy=1200, time_sell=2359).get_action(env, obs) == 0


# --- sell -------------------------------------------------------------------

def test_sell_when_close_exactly_on_sell_minute():
    env, obs = _env(_epoch(14, 0, 0))
    assert _model(time_sell=1400).get_action(env, obs) == 2


def test_sell_via_one_second_offset():
    env, obs = _env(_epoch(13, 59, 59))
    assert _model(time_sell=1400).get_action(env, obs) == 2


# --- precedence + hold ------------------------------------------------------

def test_buy_takes_precedence_when_buy_and_sell_times_coincide():
    # buy and sell both set to 12:00; buy is checked first so BUY wins.
    env, obs = _env(_epoch(12, 0, 0))
    assert _model(time_buy=1200, time_sell=1200).get_action(env, obs) == 1


def test_hold_when_no_time_matches():
    env, obs = _env(_epoch(10, 30, 0))
    assert _model(time_buy=1200, time_sell=1400).get_action(env, obs) == 0


def test_uses_last_bar_of_lookback_window():
    # lookback=2: a stale first bar at the buy time must be ignored; the recent bar at 09:00 -> HOLD.
    item_size = len(COLUMNS)
    stale = [float(_epoch(12, 0, 0)), 0.0]
    recent = [float(_epoch(9, 0, 0)), 0.0]
    obs = list(stale) + list(recent) + [99999.0]
    env = types.SimpleNamespace(
        env_config=types.SimpleNamespace(lookback_window_size=2),
        df=types.SimpleNamespace(columns=pd.Index(COLUMNS)),
    )
    assert item_size == 2
    assert _model(time_buy=1200, time_sell=1400).get_action(env, obs) == 0


def test_minute_granularity_buy_at_thirty_past():
    # time_buy 1230 -> 12:30:00 fires.
    env, obs = _env(_epoch(12, 30, 0))
    assert _model(time_buy=1230, time_sell=2359).get_action(env, obs) == 1
