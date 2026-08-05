from types import SimpleNamespace

import numpy as np
import pytest

from src.conf.env_config import EnvConfig
from src.conf.model_config import ModelConfig, ModelSupervisedConfig
from src.environment.trade_all_crypto_env import TradeAllCryptoEnv
from src.model.supervised_model import SupervisedModel


class PredictiveProvider:
    """A provider whose only feature at step t is the SIGN of the next-bar return — a perfectly
    learnable signal, so a fitted classifier should buy before up-moves. Plumbing test only: real
    data carries no such oracle."""

    def __init__(self, prices, lookback=1):
        self._prices = [float(p) for p in prices]
        self._lookback = lookback

    def get_timesteps(self):
        return len(self._prices) - 1

    def get_lookback_window(self):
        return self._lookback

    def get_price(self, step):
        i = min(max(int(step), 0), len(self._prices) - 1)
        return self._prices[i]

    def get_values(self, step):
        return np.array([1.0 if self.get_price(step + 1) > self.get_price(step) else -1.0], dtype=np.float32)

    def get_signal_buy_sell(self, step):
        return 0

    def get_signal_buy_profitable(self, step):
        return 0

    def get_signal_buy_drawdown(self, step):
        return 0


def _make_model(model_name="supervised-logreg", **kw):
    return SupervisedModel(
        ModelConfig(
            model_type="supervised",
            model_supervised=ModelSupervisedConfig(model_name=model_name, seed=0, **kw),
        )
    )


def _make_env(prices, **cfg_kwargs):
    cfg = EnvConfig(
        type="trade_all",
        initial_balance=100000,
        transaction_fee=0.0,
        observations_contain=[],
        take_profit=None,
        stop_loss=None,
        **cfg_kwargs,
    )
    env = TradeAllCryptoEnv(cfg, PredictiveProvider(prices, lookback=1), "cpu")
    env.setup("percent_profit", {})
    return env


@pytest.mark.parametrize("horizon", [2, 3, 5])
def test_forward_horizon_label_stays_inside_train_provider(horizon):
    # L7 (purge/embargo boundary): the supervised forward-horizon label reads get_price(step + horizon),
    # which must stay INSIDE the train provider (max index == timesteps, one past the last decision) — never
    # further — so a train label can never reach into the next (test) window. Pins the tail-purge implied by
    # `range(0, timesteps - horizon + 1)`; an unbounded `range(0, timesteps)` would read timesteps+horizon-1.
    prices = [100.0 + (i % 7) for i in range(50)]
    env = _make_env(prices)
    provider = env.data_provider
    reads = []
    real = provider.get_price
    provider.get_price = lambda step, _r=real, _s=reads: (_s.append(int(step)), _r(step))[1]
    _make_model(forward_horizon=horizon).train(env)
    assert reads, "no labels were built"
    assert max(reads) <= provider.get_timesteps(), (
        f"forward-horizon label read index {max(reads)} > timesteps {provider.get_timesteps()} "
        f"(horizon={horizon}) — the label reaches beyond the train provider (would leak into the next window)"
    )


def test_features_flattens_multilayer_list_and_flat_array():
    prov_list = SimpleNamespace(get_values=lambda s: [np.zeros((2, 3)), np.ones((2, 1))])
    assert SupervisedModel._features(prov_list, 0).shape == (8,)
    prov_arr = SimpleNamespace(get_values=lambda s: np.array([1.0, 2.0, 3.0]))
    assert SupervisedModel._features(prov_arr, 0).shape == (3,)


def test_action_for_long_only_mapping():
    m = _make_model()
    flat = SimpleNamespace(positions=[0], env_config=SimpleNamespace(allow_shorting=False))
    long = SimpleNamespace(positions=[5], env_config=SimpleNamespace(allow_shorting=False))
    assert m._action_for(flat, up=True) == 1
    assert m._action_for(long, up=True) == 0
    assert m._action_for(long, up=False) == 2
    assert m._action_for(flat, up=False) == 0


def test_action_for_shorting_mapping():
    m = _make_model()
    flat = SimpleNamespace(positions=[0], env_config=SimpleNamespace(allow_shorting=True))
    short = SimpleNamespace(positions=[-5], env_config=SimpleNamespace(allow_shorting=True))
    assert m._action_for(flat, up=False) == 3
    assert m._action_for(short, up=True) == 4


def test_constant_predictor_when_training_labels_one_class():
    m = _make_model()
    m._fit(np.zeros((5, 2), dtype=np.float32), np.ones(5, dtype=int))
    assert m._predict_up(np.zeros(2, dtype=np.float32)) is True


def test_does_not_produce_an_sb3_checkpoint():
    assert _make_model().produces_checkpoint() is False


@pytest.mark.parametrize("model_name", ["supervised-logreg", "supervised-gbm"])
def test_learns_predictable_pattern_and_profits(model_name):
    # A long series so the tree baseline clears its default min_samples_leaf (real runs have 1000s of bars).
    prices = [100, 102, 100, 103, 101, 104, 102, 105, 103, 106, 104, 107] * 40
    m = _make_model(model_name)
    m.train(_make_env(prices))
    env_test = _make_env(prices)
    m.test(env_test, True)
    state = env_test.get_run_state()
    assert state[17] >= 1  # completed at least one round-trip
    # The env resets stake to `initial` after each closed trade, so per-trade P&L accrues into
    # total_profit (the summary reconstructs the compounded curve) — assert that, not net_worths[-1].
    assert env_test.total_profit > 0
