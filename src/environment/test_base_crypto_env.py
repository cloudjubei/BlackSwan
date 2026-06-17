"""Direct tests for the additive ``tpsl_kinds`` exit-reason tag on BaseCryptoEnv.

These exercise ``resolve_tpsl`` / ``resolve_action`` in isolation via ``__new__`` (bypassing the
heavy data-provider __init__), so they assert ONLY the tag wiring without changing trading behaviour.
Plain asserts (no pytest dependency) so they run under the project venv, which has no pytest.
"""

import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.environment.base_crypto_env import BaseCryptoEnv


def _env(position, net_worth, *, entry=100.0, price=130.0, highest=140.0, lowest=0.0,
         take_profit=0.1, trailing=0.05, stop_loss=0.1):
    e = BaseCryptoEnv.__new__(BaseCryptoEnv)
    e.env_config = types.SimpleNamespace(
        take_profit=take_profit, trailing_take_profit=trailing, stop_loss=stop_loss
    )
    e.positions = [position]
    e.initial_net_worth = 100000.0
    e.current_step = 0
    e.current_price = price
    e.position_price_entry = entry
    e.position_price_highest = highest
    e.position_price_lowest = lowest
    e._calculate_net_worth = lambda step: net_worth
    return e


def test_resolve_tpsl_flat_returns_none_triplet():
    forced, tp, kind = _env(0, 100000.0).resolve_tpsl()
    assert (forced, tp, kind) == (0, None, None)


def test_resolve_tpsl_trailing_kind():
    # long, profit 5% (< take_profit 10% so not a plain TP); highest 140 >= activation 110 and
    # price 130 <= 140*(1-0.05)=133 -> trailing stop fires.
    forced, tp, kind = _env(10.0, 105000.0, highest=140.0, price=130.0).resolve_tpsl()
    assert forced == 2 and tp is True and kind == "trailing"


def test_resolve_tpsl_regular_take_profit_kind():
    # highest below activation so no trailing; net worth +12% >= take_profit 10% -> plain tp.
    forced, tp, kind = _env(10.0, 112000.0, highest=105.0).resolve_tpsl()
    assert forced == 2 and tp is True and kind == "tp"


def test_resolve_tpsl_stop_loss_kind():
    # no take-profit hit; loss 15% >= stop_loss 10% -> sl.
    forced, tp, kind = _env(10.0, 85000.0, highest=105.0).resolve_tpsl()
    assert forced == 2 and tp is False and kind == "sl"


def test_resolve_tpsl_short_close_action_is_4():
    # short position closes/covers with action 4; trailing mirrored on the lows.
    forced, tp, kind = _env(-10.0, 105000.0, entry=100.0, price=80.0, lowest=70.0).resolve_tpsl()
    assert forced == 4 and tp is True and kind == "trailing"


def _action_env():
    e = BaseCryptoEnv.__new__(BaseCryptoEnv)
    e.env_config = types.SimpleNamespace(take_profit=0.1, trailing_take_profit=0.05, stop_loss=0.1)
    e.positions = [10.0]
    e.initial_net_worth = 100000.0
    e.current_step = 0
    e.current_price = 130.0
    e.position_price_entry = 100.0
    e.position_price_highest = 140.0
    e.position_price_lowest = 0.0
    e._calculate_net_worth = lambda step: 105000.0
    e.tpsls, e.tpsl_kinds, e.actions, e.actions_made, e.forced_actions = [], [], [], [], []
    return e


def test_resolve_action_records_forced_trailing_kind():
    e = _action_env()
    e.take_action = lambda action: action in (2, 4)  # only a close executes
    e.resolve_action(0)  # agent holds; trailing stop forces a close
    assert e.tpsls == [1]
    assert e.tpsl_kinds == ["trailing"]
    assert e.forced_actions == [2]
    assert e.actions_made == [True]


def test_resolve_action_agent_close_records_none_kind():
    e = _action_env()
    e.take_action = lambda action: True  # the agent's own action executes
    e.resolve_action(2)  # agent closes the long itself; no forced TP/SL
    assert e.tpsls == [0]
    assert e.tpsl_kinds == [None]
    assert e.forced_actions == [0]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    return len(fns)


if __name__ == "__main__":
    print(f"{_run_all()} env tag tests passed")
