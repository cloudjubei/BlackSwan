"""Forward-pass / predict-branch correctness for the IQN network and algorithm.

Covers:
  * CosineEmbeddingNetwork.forward shape (batch, num_tau, features_dim).
  * QuantileNetwork.forward quantile-tensor shape (batch, num_tau, num_actions) and _predict
    greedy-action shape.
  * IQN.predict's epsilon-greedy exploration branch (vectorized + single obs) and the deterministic
    delegation branch — exercised via __new__ so we never stand up the full off-policy algorithm.

Tiny dims, CPU only. No replay buffer, no training loop, no GPU.
"""

import types

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import FlattenExtractor

from src.model.iqn.iqn import IQN
from src.model.iqn.policies import CosineEmbeddingNetwork, QuantileNetwork


def _box(dim):
    return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def _quantile_net(obs_dim=6, n_actions=3, n_quantiles=5, num_cosine=7, net_arch=None):
    obs_space = _box(obs_dim)
    act_space = spaces.Discrete(n_actions)
    fe = FlattenExtractor(obs_space)
    return QuantileNetwork(
        obs_space,
        act_space,
        fe,
        fe.features_dim,
        n_quantiles=n_quantiles,
        num_cosine=num_cosine,
        net_arch=net_arch or [8],
    )


# --- CosineEmbeddingNetwork -------------------------------------------------


@pytest.mark.parametrize("batch,num_tau,features_dim", [(4, 5, 9), (1, 3, 4), (2, 8, 6)])
def test_cosine_embedding_shape(batch, num_tau, features_dim):
    ce = CosineEmbeddingNetwork(num_cosine=7, features_dim=features_dim)
    out = ce(th.rand(batch, num_tau))
    assert tuple(out.shape) == (batch, num_tau, features_dim)
    assert th.isfinite(out).all()


def test_cosine_embedding_nonnegative_after_relu():
    # The net ends in a ReLU, so all embedding entries are >= 0.
    ce = CosineEmbeddingNetwork(num_cosine=5, features_dim=6)
    out = ce(th.rand(3, 4))
    assert (out >= 0).all()


# --- QuantileNetwork.forward ------------------------------------------------


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("num_tau", [3, 6])
def test_quantile_forward_shape(batch, num_tau):
    net = _quantile_net(obs_dim=6, n_actions=3, n_quantiles=5)
    out = net(th.zeros(batch, 6), num_tau)
    # (batch, num_tau_samples, num_actions)
    assert tuple(out.shape) == (batch, num_tau, 3)
    assert th.isfinite(out).all()


def test_quantile_forward_respects_action_dim():
    net = _quantile_net(obs_dim=6, n_actions=7, n_quantiles=4)
    out = net(th.zeros(2, 6), 4)
    assert out.shape[-1] == 7


def test_quantile_predict_returns_one_greedy_action_per_row():
    net = _quantile_net(obs_dim=6, n_actions=3, n_quantiles=5)
    actions = net._predict(th.zeros(4, 6))
    assert tuple(actions.shape) == (4,)
    assert actions.dtype == th.int64
    assert int(actions.min()) >= 0 and int(actions.max()) <= 2


# --- IQN.predict (epsilon-greedy + delegation) ------------------------------


def _bare_iqn(exploration_rate, n_actions=3, obs_dim=4):
    # __new__ bypasses the heavy OffPolicyAlgorithm init; predict() only reads these attributes.
    m = IQN.__new__(IQN)
    m.exploration_rate = exploration_rate
    m.observation_space = _box(obs_dim)
    m.action_space = spaces.Discrete(n_actions)
    return m


def test_predict_exploration_vectorized_returns_one_action_per_batch():
    m = _bare_iqn(exploration_rate=1.0)  # always explore
    np.random.seed(0)
    obs = np.zeros((2, 4), dtype=np.float32)
    action, state = m.predict(obs, deterministic=False)
    assert action.shape == (2,)
    assert state is None
    assert set(np.unique(action)).issubset({0, 1, 2})


def test_predict_exploration_single_obs_returns_scalar_action():
    m = _bare_iqn(exploration_rate=1.0)
    np.random.seed(1)
    obs = np.zeros((4,), dtype=np.float32)
    action, _ = m.predict(obs, deterministic=False)
    assert action.shape == ()
    assert int(action) in (0, 1, 2)


def test_predict_deterministic_delegates_to_policy():
    # deterministic=True must skip exploration entirely and return the policy's action/state.
    m = _bare_iqn(exploration_rate=1.0)
    sentinel = (np.array([2]), "HIDDEN")
    m.policy = types.SimpleNamespace(predict=lambda o, s, e, d: sentinel)
    action, state = m.predict(np.zeros((1, 4), dtype=np.float32), deterministic=True)
    assert int(action[0]) == 2 and state == "HIDDEN"


def test_predict_low_exploration_rate_delegates_to_policy():
    # exploration_rate 0 -> rand() (>=0) never < 0, so always delegate even when non-deterministic.
    m = _bare_iqn(exploration_rate=0.0)
    sentinel = (np.array([1]), None)
    m.policy = types.SimpleNamespace(predict=lambda o, s, e, d: sentinel)
    action, _ = m.predict(np.zeros((1, 4), dtype=np.float32), deterministic=False)
    assert int(action[0]) == 1


def test_create_aliases_wires_quantile_nets_from_policy():
    # _create_aliases is pure attribute plumbing off self.policy; verify via __new__.
    m = IQN.__new__(IQN)
    m.policy = types.SimpleNamespace(quantile_net="QN", quantile_net_target="QNT", n_quantiles=11)
    m._create_aliases()
    assert m.quantile_net == "QN"
    assert m.quantile_net_target == "QNT"
    assert m.n_quantiles == 11


def test_excluded_save_params_drops_quantile_nets():
    m = IQN.__new__(IQN)
    # Stub the super() call's contribution by patching the bound method's reliance: call directly.
    params = IQN._excluded_save_params(m)
    assert "quantile_net" in params and "quantile_net_target" in params


def test_get_torch_save_params_lists_policy_and_optimizer():
    m = IQN.__new__(IQN)
    state_dicts, others = IQN._get_torch_save_params(m)
    assert state_dicts == ["policy", "policy.optimizer"]
    assert others == []
