import numpy as np
import pytest

from src.conf.env_config import EnvConfig
from src.environment.trade_all_crypto_env import TradeAllCryptoEnv

_MULTIPLIERS = {
    "combo_sell": 1.0,
    "combo_buy": 1.0,
    "combo_positionprofitpercentage": 1.0,
    "combo_wrongaction": -1.0,
    "combo_noaction": 0.0,
    "combo_fee_penalty": 1.0,
}


class FakeProvider:
    """Minimal data provider: a flat price series + zeroed features/signals, enough to drive the env."""

    def __init__(self, prices, lookback=1, features=2):
        self._prices = [float(p) for p in prices]
        self._lookback = lookback
        self._features = features

    def get_timesteps(self):
        return len(self._prices) - 1

    def get_lookback_window(self):
        return self._lookback

    def get_price(self, step):
        i = min(max(int(step), 0), len(self._prices) - 1)
        return self._prices[i]

    def get_values(self, step):
        return np.zeros(self._features, dtype=np.float32)

    def get_signal_buy_sell(self, step):
        return 0

    def get_signal_buy_profitable(self, step):
        return 0

    def get_signal_buy_drawdown(self, step):
        return 0


def make_env(prices, transaction_fee=0.0, **cfg_kwargs):
    cfg = EnvConfig(
        type="trade_all",
        initial_balance=100000,
        transaction_fee=transaction_fee,
        observations_contain=[],
        **cfg_kwargs,
    )
    env = TradeAllCryptoEnv(cfg, FakeProvider(prices, lookback=1), "cpu")
    env.reward_model = "combo_all"
    env.reward_multipliers = dict(_MULTIPLIERS)
    return env


def _arm(env, step, price, balance=100000, position=0):
    env.current_step = step
    env.current_price = price
    env.balances[-1] = balance
    env.positions[-1] = position


# --- vol-targeted position sizing ---


def test_position_size_fixed_is_all_in():
    env = make_env([100] * 20)
    env.current_step = 12
    assert env._position_size() == 1.0


def test_realized_vol_matches_numpy_std():
    prices = [100, 101, 102, 101, 103, 104, 102, 105, 106, 104, 107, 108, 106]
    env = make_env(prices, position_sizing="vol_target", vol_window=10)
    env.current_step = 12
    sub = prices[12 - 10 : 13]
    rets = [sub[i] / sub[i - 1] - 1 for i in range(1, len(sub))]
    assert env._realized_vol() == pytest.approx(float(np.std(rets)))


def test_position_size_caps_at_one_in_calm_regime():
    env = make_env(
        [100 + 0.001 * i for i in range(20)],
        position_sizing="vol_target",
        vol_target=0.02,
        vol_target_min=0.1,
        vol_window=10,
    )
    env.current_step = 15
    assert env._position_size() == 1.0


def test_position_size_shrinks_in_volatile_regime():
    env = make_env(
        [100 * (1.06 if i % 2 else 0.94) for i in range(20)],
        position_sizing="vol_target",
        vol_target=0.02,
        vol_target_min=0.1,
        vol_window=10,
    )
    env.current_step = 15
    size = env._position_size()
    assert 0.1 <= size < 1.0


def test_take_action_buy_fixed_is_all_in():
    env = make_env([100] * 20)
    _arm(env, 5, 100)
    assert env.take_action(1) is True
    assert env.positions[-1] == pytest.approx(1000)
    assert env.balances[-1] == 0


def test_take_action_buy_vol_target_deploys_a_fraction(monkeypatch):
    env = make_env([100] * 20, position_sizing="vol_target")
    _arm(env, 12, 100)
    monkeypatch.setattr(env, "_position_size", lambda: 0.5)
    assert env.take_action(1) is True
    assert env.positions[-1] == pytest.approx(500)
    assert env.balances[-1] == pytest.approx(50000)


# --- shorting ---


def _short_env(prices, **kw):
    env = make_env(prices, allow_shorting=True, transaction_fee=0.0, **kw)
    env.setup("combo_all", dict(_MULTIPLIERS))
    return env


def test_action_space_is_discrete_5_with_shorting():
    from gymnasium import spaces

    env = make_env([100] * 8, allow_shorting=True)
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 5


def test_short_then_cover_profits_when_price_falls():
    env = _short_env([100, 100, 90, 90, 90, 90])
    env.step(3)  # short @100
    env.step(0)  # hold
    env.step(4)  # cover @90
    state = env.get_run_state()
    assert state[17] == 1
    assert state[1] > 0


def test_short_then_cover_loses_when_price_rises():
    env = _short_env([100, 100, 110, 110, 110, 110])
    env.step(3)
    env.step(0)
    env.step(4)
    state = env.get_run_state()
    assert state[17] == 1
    assert state[1] < 0


def test_short_stop_loss_forces_cover_when_price_rises():
    env = _short_env([100, 100, 103, 103, 103, 103], take_profit=0.10, stop_loss=0.02)
    env.step(3)
    env.step(0)
    env.step(0)  # @103 -> 3% loss >= SL 2% -> forced cover
    state = env.get_run_state()
    assert state[17] == 1
    assert len(env.trades_sl) == 1


def test_short_take_profit_forces_cover_when_price_falls():
    env = _short_env([100, 100, 97, 97, 97, 97], take_profit=0.02, stop_loss=0.10)
    env.step(3)
    env.step(0)
    env.step(0)  # @97 -> 3% profit >= TP 2% -> forced cover
    state = env.get_run_state()
    assert state[17] == 1
    assert len(env.trades_tp) == 1


def test_long_path_unaffected_by_shorting_enabled():
    env = _short_env([100, 100, 110, 110, 110, 110])
    env.step(1)  # long @100
    env.step(0)
    env.step(2)  # close @110
    state = env.get_run_state()
    assert state[17] == 1
    assert state[1] > 0


def test_short_action_ignored_when_shorting_disabled():
    env = make_env([100] * 8, transaction_fee=0.0)
    env.setup("combo_all", dict(_MULTIPLIERS))
    env.step(3)
    assert env.positions[-1] == 0


# --- long-only regression guards for the sign-aware refactor ---


def test_long_stop_loss_forces_close_when_price_falls():
    env = make_env([100, 100, 97, 97, 97, 97], take_profit=0.10, stop_loss=0.02)
    env.setup("combo_all", dict(_MULTIPLIERS))
    env.step(1)
    env.step(0)
    env.step(0)  # @97 -> 3% loss >= SL 2% -> forced sell
    state = env.get_run_state()
    assert state[17] == 1
    assert len(env.trades_sl) == 1


def test_long_take_profit_forces_close_when_price_rises():
    env = make_env([100, 100, 103, 103, 103, 103], take_profit=0.02, stop_loss=0.10)
    env.setup("combo_all", dict(_MULTIPLIERS))
    env.step(1)
    env.step(0)
    env.step(0)  # @103 -> 3% profit >= TP 2% -> forced sell
    state = env.get_run_state()
    assert state[17] == 1
    assert len(env.trades_tp) == 1


# --- direct / recurrent reward variants ---


def test_profit_percentage_direct_rewards_step_return():
    env = make_env([100, 110, 110, 110], transaction_fee=0.0)
    env.setup("profit_percentage_direct", dict(_MULTIPLIERS))
    env.step(1)  # long @100
    _, reward, _, _, _ = env.step(0)  # @110 -> +10% step return
    assert reward == pytest.approx(0.1, abs=1e-6)


def test_differential_sharpe_is_finite_and_nonzero():
    env = make_env([100, 110, 105, 115, 108, 120, 112], transaction_fee=0.0)
    env.setup("differential_sharpe", dict(_MULTIPLIERS))
    env.step(1)
    rewards = [env.step(0)[1] for _ in range(4)]
    assert all(np.isfinite(r) for r in rewards)
    assert any(abs(r) > 0 for r in rewards)


# --- no-op action penalty (combo_all_noop) ---


def _noop_env(reward_model="combo_all_noop"):
    env = make_env([100] * 8)
    env.reward_model = reward_model
    env.reward_multipliers = dict(_MULTIPLIERS)
    return env


def _set_last(env, action, made, forced=0):
    env.actions = [action]
    env.actions_made = [made]
    env.forced_actions = [forced]


def test_noop_penalty_penalizes_unexecuted_buy():
    env = _noop_env()
    _set_last(env, 1, False, 0)  # tried to buy, nothing executed (already long / no cash)
    assert env._noop_penalty() > 0


def test_noop_penalty_zero_for_executed_action():
    env = _noop_env()
    _set_last(env, 1, True, 0)  # the agent's buy actually executed
    assert env._noop_penalty() == 0.0


def test_noop_penalty_zero_on_hold():
    env = _noop_env()
    _set_last(env, 0, False, 0)
    assert env._noop_penalty() == 0.0


def test_noop_penalty_fires_when_only_a_forced_tpsl_executed():
    env = _noop_env()
    _set_last(env, 1, True, 2)  # agent's buy was a no-op; a forced TP/SL (2) did the close
    assert env._noop_penalty() > 0


def test_noop_penalty_off_for_other_reward_models():
    env = _noop_env("combo_all")
    _set_last(env, 1, False, 0)
    assert env._noop_penalty() == 0.0


def test_noop_penalty_uses_configured_value():
    env = _noop_env()
    _set_last(env, 1, False, 0)
    env.reward_multipliers = {**_MULTIPLIERS, "combo_noop_penalty": 0.02}
    assert env._noop_penalty() == pytest.approx(0.02)


def test_noop_penalty_off_when_value_zero():
    env = _noop_env()
    _set_last(env, 1, False, 0)
    env.reward_multipliers = {**_MULTIPLIERS, "combo_noop_penalty": 0.0}
    assert env._noop_penalty() == 0.0


def test_combo_all_noop_lowers_reward_on_a_noop_buy_vs_combo_all():
    # buy, then buy AGAIN (a no-op while already long): penalized under combo_all_noop, not combo_all.
    noop = make_env([100, 100, 100, 100])
    noop.setup("combo_all_noop", dict(_MULTIPLIERS))
    noop.step(1)
    r_noop = noop.step(1)[1]
    plain = make_env([100, 100, 100, 100])
    plain.setup("combo_all", dict(_MULTIPLIERS))
    plain.step(1)
    r_plain = plain.step(1)[1]
    assert r_noop < r_plain
