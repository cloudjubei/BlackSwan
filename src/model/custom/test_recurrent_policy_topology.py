"""Speedup proofs for the LSTM-topology levers on CustomRecurrentActorCriticPolicy.

A4 (shared_lstm) and A5 (lstm_hidden_size) only buy speed if they genuinely SHRINK the recurrent
policy's parameter + per-step compute footprint. These construct the policy directly (no training) and
assert the parameter count actually drops — a deterministic proof that the cheaper topology is real, not
merely accepted by the config.
"""

import numpy as np
from gymnasium import spaces

from src.model.custom.policies import CustomRecurrentActorCriticPolicy


def _param_count(**kw):
    obs = spaces.Box(low=-1, high=1, shape=(32,), dtype=np.float32)
    policy = CustomRecurrentActorCriticPolicy(
        obs, spaces.Discrete(3), lambda _: 1e-3,
        net_arch=[64], custom_net_arch=["Linear", "Linear"], **kw,
    )
    return sum(t.numel() for t in policy.parameters())


def test_shared_lstm_has_fewer_params_than_separate():
    separate = _param_count(lstm_hidden_size=256, shared_lstm=False, enable_critic_lstm=True)
    shared = _param_count(lstm_hidden_size=256, shared_lstm=True, enable_critic_lstm=False)
    assert shared < separate  # one shared LSTM instead of two -> the critic LSTM's params/compute are gone


def test_smaller_lstm_hidden_size_has_fewer_params():
    big = _param_count(lstm_hidden_size=256, shared_lstm=False, enable_critic_lstm=True)
    small = _param_count(lstm_hidden_size=64, shared_lstm=False, enable_critic_lstm=True)
    assert small < big  # a narrower LSTM is strictly cheaper per step
