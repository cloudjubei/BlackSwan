"""Regression tests for RRLModel — the continuous-position -> discrete-action state machine (the trickiest
bit) and that train (Sharpe gradient ascent) + test (stepping the env) run a full episode without SB3.
Hermetic: a tiny fake env + data provider, no real klines."""
from types import SimpleNamespace

import numpy as np

from src.conf.model_config import ModelConfig, ModelRLConfig
from src.model.custom.rrl.rrl_model import RRLModel


def test_action_for_maps_target_position_to_the_right_transition():
    m = RRLModel.__new__(RRLModel)  # pure state machine — no __init__ needed
    flat, long, short = SimpleNamespace(positions=[0]), SimpleNamespace(positions=[2.5]), SimpleNamespace(positions=[-1.0])
    # from flat
    assert m._action_for(flat, 1) == 1   # open long
    assert m._action_for(flat, -1) == 3  # open short
    assert m._action_for(flat, 0) == 0   # hold
    # from long
    assert m._action_for(long, 1) == 0   # hold
    assert m._action_for(long, 0) == 2   # close long
    assert m._action_for(long, -1) == 2  # close first (the flip completes next step)
    # from short
    assert m._action_for(short, -1) == 0  # hold
    assert m._action_for(short, 1) == 4   # cover
    assert m._action_for(short, 0) == 4   # cover


class _DP:
    def __init__(self, steps, dim):
        rng = np.random.default_rng(0)
        self._x = rng.standard_normal((steps, dim)).astype(np.float32)
        self._p = (100.0 + np.cumsum(rng.standard_normal(steps))).astype(float)
        self._steps = steps

    def get_timesteps(self):
        return self._steps

    def get_values(self, t):
        return self._x[min(t, self._steps - 1)]

    def get_price(self, t):
        return float(self._p[min(t, self._steps - 1)])

    def get_lookback_window(self):
        return 1


class _Env:
    def __init__(self, allow_short, steps=60, dim=4):
        self.data_provider = _DP(steps, dim)
        self.env_config = SimpleNamespace(allow_shorting=allow_short, transaction_fee=0.001)
        self.positions = [0]
        self.current_step = 0
        self._steps = steps
        self.actions = []

    def reset(self):
        self.current_step = 0
        self.positions = [0]
        self.actions = []
        return None, {}

    def step(self, action):
        self.actions.append(action)
        pos = self.positions[-1]
        nxt = pos
        if action == 1 and pos == 0:
            nxt = 1.0
        elif action == 2 and pos > 0:
            nxt = 0
        elif action == 3 and pos == 0:
            nxt = -1.0
        elif action == 4 and pos < 0:
            nxt = 0
        self.positions.append(nxt)
        self.current_step += 1
        return None, 0.0, self.current_step >= self._steps, False, {}


def _model(env, episodes=2):
    rl = ModelRLConfig(
        model_name="rrl", reward_model="combo_unified", net_arch=[64, 32], custom_net_arch=[],
        optimizer_class="Adam", activation_fn="ReLU", learning_rate=0.05, batch_size=16, gamma=0.99,
        seed=None, episodes=episodes,
    )
    return RRLModel(ModelConfig(model_type="rl", model_rl=rl), env, "cpu")


def test_train_then_test_runs_a_full_episode():
    env = _Env(allow_short=False, steps=60)
    m = _model(env)
    m.train(env)  # Sharpe gradient ascent — must not crash
    m.test(env)
    assert env.current_step == env._steps  # stepped the whole test episode


def test_long_only_env_never_emits_short_or_cover_actions():
    env = _Env(allow_short=False, steps=50)
    m = _model(env)
    m.train(env)
    m.test(env)
    assert 3 not in env.actions and 4 not in env.actions  # no short/cover when shorting is disabled


def test_rrl_never_resumes_a_checkpoint():
    m = _model(_Env(allow_short=True, steps=20))
    assert m.is_pretrained() is False and m.produces_checkpoint() is False
