"""Direct unit tests for the deterministic TimeStrategyModel baseline.

get_action reads the current bar's UTC hour from ``env.data_provider.get_timestamp(env.current_step)`` and
opens a long at ``time_buy`` (hour 0-23), closes at ``time_sell``, else holds. We feed a fake provider whose
get_timestamp returns a bar at a chosen UTC hour and assert the emitted action (1 buy / 2 sell / 0 hold).
"""
from types import SimpleNamespace

import pandas as pd

from src.conf.model_config import ModelConfig, ModelTimeConfig
from src.model.time_strategy_model import TimeStrategyModel


def _model(time_buy, time_sell):
    return TimeStrategyModel(
        ModelConfig(model_type="time", model_time=ModelTimeConfig(time_buy=time_buy, time_sell=time_sell))
    )


def _env(hour):
    """A fake env whose CURRENT bar sits at the given UTC hour."""
    dt = pd.Timestamp(2021, 1, 1, hour, 0, 0, tz="UTC")
    return SimpleNamespace(current_step=0, data_provider=SimpleNamespace(get_timestamp=lambda _s: dt))


def test_opens_long_at_the_buy_hour():
    assert _model(14, 21).get_action(_env(14), None) == 1


def test_closes_long_at_the_sell_hour():
    assert _model(14, 21).get_action(_env(21), None) == 2


def test_holds_at_every_other_hour():
    m = _model(14, 21)
    for h in (0, 1, 13, 15, 20, 22, 23):
        assert m.get_action(_env(h), None) == 0


def test_overnight_config_buys_late_sells_early_and_holds_through_the_night():
    # buy 21, sell 14 (next day) -> an overnight hold of the off-US-session window.
    m = _model(21, 14)
    assert m.get_action(_env(21), None) == 1  # open at night
    assert m.get_action(_env(14), None) == 2  # close mid-next-day
    assert m.get_action(_env(3), None) == 0  # hold through the night


def test_id_encodes_the_two_hours():
    mid = _model(14, 21).id
    assert "time" in mid and "14" in mid and "21" in mid
