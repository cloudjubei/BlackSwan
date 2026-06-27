"""Unit tests for the deterministic DayOfWeekStrategyModel baseline.

get_action reads the current bar's weekday from ``env.data_provider.get_timestamp(env.current_step)`` and
opens a long on ``day_buy`` (0=Mon..6=Sun), closes on ``day_sell``, else holds. We feed a fake provider
whose get_timestamp lands on a chosen weekday and assert the emitted action (1 buy / 2 sell / 0 hold).
"""
from types import SimpleNamespace

import pandas as pd

from src.conf.model_config import ModelConfig, ModelDayConfig
from src.model.day_of_week_strategy_model import DayOfWeekStrategyModel


def _model(day_buy, day_sell):
    return DayOfWeekStrategyModel(
        ModelConfig(model_type="weekday", model_day=ModelDayConfig(day_buy=day_buy, day_sell=day_sell))
    )


def _env(weekday):
    # 2021-01-04 is a Monday (weekday 0); offset by `weekday` days to land on that weekday.
    dt = pd.Timestamp(2021, 1, 4, 12, 0, 0, tz="UTC") + pd.Timedelta(days=weekday)
    assert dt.weekday() == weekday
    return SimpleNamespace(current_step=0, data_provider=SimpleNamespace(get_timestamp=lambda _s: dt))


def test_opens_long_on_the_buy_day():
    assert _model(0, 4).get_action(_env(0), None) == 1  # Monday


def test_closes_long_on_the_sell_day():
    assert _model(0, 4).get_action(_env(4), None) == 2  # Friday


def test_holds_on_every_other_day():
    m = _model(0, 4)
    for wd in (1, 2, 3, 5, 6):
        assert m.get_action(_env(wd), None) == 0


def test_weekend_hold_buys_friday_sells_monday():
    m = _model(4, 0)  # buy Fri, sell Mon -> hold across the weekend (crypto trades Sat/Sun)
    assert m.get_action(_env(4), None) == 1  # Friday open
    assert m.get_action(_env(5), None) == 0  # Saturday hold
    assert m.get_action(_env(6), None) == 0  # Sunday hold
    assert m.get_action(_env(0), None) == 2  # Monday close


def test_id_encodes_the_two_days():
    mid = _model(0, 4).id
    assert "weekday" in mid and "0" in mid and "4" in mid
