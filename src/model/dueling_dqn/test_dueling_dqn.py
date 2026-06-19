"""Forward-pass correctness for Dueling DQN: the DuelingQNetwork value/advantage aggregation and
that a DuelingDQN model wires the dueling Q-net as its policy network.

Tiny CPU dims. DuelingQNetwork builds standalone with a FlattenExtractor (no algorithm/env). The
DuelingDQN model is built once against a 5-step toy gym env (no training) to confirm the policy
uses the dueling network and predict() returns a valid discrete action.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import FlattenExtractor

from src.model.dueling_dqn.policies import DuelingDQNPolicy, DuelingQNetwork


def _box(dim):
    return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def _dueling_net(obs_dim=6, n_actions=3, net_arch=None):
    obs_space = _box(obs_dim)
    act_space = spaces.Discrete(n_actions)
    fe = FlattenExtractor(obs_space)
    return DuelingQNetwork(obs_space, act_space, fe, fe.features_dim, net_arch=net_arch or [8, 8])


# --- DuelingQNetwork.forward ------------------------------------------------


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("n_actions", [2, 3, 5])
def test_forward_shape_is_batch_by_actions(batch, n_actions):
    net = _dueling_net(obs_dim=6, n_actions=n_actions)
    out = net(th.zeros(batch, 6))
    assert tuple(out.shape) == (batch, n_actions)
    assert th.isfinite(out).all()


def test_value_and_advantage_streams_have_expected_output_dims():
    # Dueling decomposition: a scalar value stream (out 1) and a per-action advantage stream.
    net = _dueling_net(obs_dim=6, n_actions=4)
    val_last = [m for m in net.value_stream if isinstance(m, th.nn.Linear)][-1]
    adv_last = [m for m in net.advantage_stream if isinstance(m, th.nn.Linear)][-1]
    assert val_last.out_features == 1
    assert adv_last.out_features == 4


def test_qvals_equal_value_plus_centered_advantage():
    # forward computes qvals = value + (advantage - advantage.mean()); recompute the streams
    # directly and assert the aggregation matches exactly.
    th.manual_seed(0)
    net = _dueling_net(obs_dim=6, n_actions=3)
    net.eval()
    x = th.randn(4, 6)
    with th.no_grad():
        feats = net.extract_features(x, net.features_extractor)
        values = net.value_stream(feats)
        advantages = net.advantage_stream(feats)
        expected = values + (advantages - advantages.mean())
        got = net(x)
    assert th.allclose(got, expected, atol=1e-6)


# --- DuelingDQN model wiring (toy env, no training) -------------------------


class _TinyEnv(gym.Env):
    """Minimal gymnasium env so SB3 can build a DuelingDQN without real market data."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = _box(4)
        self.action_space = spaces.Discrete(3)
        self._t = 0
        self.render_mode = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        return np.zeros(4, dtype=np.float32), 1.0, self._t >= 5, False, {}


def test_dueling_dqn_uses_dueling_qnetwork_and_predicts():
    from src.model.dueling_dqn.dueling_dqn import DuelingDQN

    model = DuelingDQN(
        _TinyEnv(),
        DuelingDQNPolicy,
        buffer_size=100,
        learning_starts=1,
        batch_size=4,
        policy_kwargs=dict(net_arch=[8, 8]),
        device="cpu",
    )
    assert isinstance(model.q_net, DuelingQNetwork)
    assert isinstance(model.q_net_target, DuelingQNetwork)
    action, _ = model.predict(np.zeros(4, dtype=np.float32), deterministic=True)
    assert int(action) in (0, 1, 2)
