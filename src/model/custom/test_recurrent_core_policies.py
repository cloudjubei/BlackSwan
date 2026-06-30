"""Tests for the GRU / S4D recurrent actor-critic policies.

These build the policies directly (gym spaces + a constant lr schedule, mirroring
test_recurrent_policy_topology.py) and also run a short RecurrentPPO.learn on a tiny env — the
end-to-end proof that each non-LSTM core satisfies the rollout-buffer / collect / train contract.
"""

import numpy as np
import pytest
import torch as th
from gymnasium import spaces, Env
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.torch_layers import MlpExtractor

from src.model.custom.custommlpextractor import CustomMlpExtractor
from src.model.custom.recurrent_cores import GRURecurrentCore, S4DRecurrentCore
from src.model.custom.recurrent_core_policies import (
    RecurrentCoreActorCriticPolicy,
    GRURecurrentActorCriticPolicy,
    CustomGRURecurrentActorCriticPolicy,
    S4DRecurrentActorCriticPolicy,
    CustomS4DRecurrentActorCriticPolicy,
)
from torch import nn as _nn

OBS = spaces.Box(low=-1, high=1, shape=(32,), dtype=np.float32)
ACT = spaces.Discrete(3)


def _make(policy_cls, **kw):
    kw.setdefault("net_arch", [16])
    kw.setdefault("custom_net_arch", ["Linear", "Linear"])
    kw.setdefault("lstm_hidden_size", 8)
    return policy_cls(OBS, ACT, lambda _: 1e-3, **kw)


def _zero_states(policy, batch):
    core = policy.lstm_actor
    shape = (core.num_layers, batch, core.hidden_size)
    return RNNStates((th.zeros(shape), th.zeros(shape)), (th.zeros(shape), th.zeros(shape)))


class _TinyEnv(Env):
    """Minimal gymnasium env so RecurrentPPO can run a couple of rollout/train cycles."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.render_mode = None
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self._t += 1
        return self.observation_space.sample(), 1.0, self._t >= 5, False, {}


ALL_POLICIES = [
    GRURecurrentActorCriticPolicy,
    CustomGRURecurrentActorCriticPolicy,
    S4DRecurrentActorCriticPolicy,
    CustomS4DRecurrentActorCriticPolicy,
]
GRU_POLICIES = [GRURecurrentActorCriticPolicy, CustomGRURecurrentActorCriticPolicy]
S4D_POLICIES = [S4DRecurrentActorCriticPolicy, CustomS4DRecurrentActorCriticPolicy]
CUSTOM_POLICIES = [CustomGRURecurrentActorCriticPolicy, CustomS4DRecurrentActorCriticPolicy]
PLAIN_POLICIES = [GRURecurrentActorCriticPolicy, S4DRecurrentActorCriticPolicy]


# --------------------------------------------------------------------------------------
# Construction: right core, right head, decoupled output/state widths
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("policy_cls", GRU_POLICIES)
def test_gru_policy_uses_gru_core(policy_cls):
    policy = _make(policy_cls)
    assert isinstance(policy.lstm_actor, GRURecurrentCore)


@pytest.mark.parametrize("policy_cls", S4D_POLICIES)
def test_s4d_policy_uses_s4d_core(policy_cls):
    policy = _make(policy_cls)
    assert isinstance(policy.lstm_actor, S4DRecurrentCore)


@pytest.mark.parametrize("policy_cls", CUSTOM_POLICIES)
def test_custom_variants_use_custom_mlp_head(policy_cls):
    policy = _make(policy_cls)
    assert isinstance(policy.mlp_extractor, CustomMlpExtractor)


@pytest.mark.parametrize("policy_cls", PLAIN_POLICIES)
def test_plain_variants_use_default_mlp_head(policy_cls):
    policy = _make(policy_cls)
    assert isinstance(policy.mlp_extractor, MlpExtractor)
    assert not isinstance(policy.mlp_extractor, CustomMlpExtractor)


def test_gru_output_width_equals_state_width():
    policy = _make(GRURecurrentActorCriticPolicy, lstm_hidden_size=8)
    assert policy.lstm_output_dim == 8
    assert policy.lstm_actor.hidden_size == 8  # GRU: one hidden vector per channel


def test_s4d_decouples_output_width_from_packed_state_width():
    policy = _make(S4DRecurrentActorCriticPolicy, lstm_hidden_size=8, ssm_state_dim=16)
    # Per-step output is one value per channel; threaded state is channels x state_dim, split Re/Im.
    assert policy.lstm_output_dim == 8
    assert policy.lstm_actor.out_features == 8
    assert policy.lstm_actor.hidden_size == 8 * 16


# --------------------------------------------------------------------------------------
# shared / separate critic core wiring (mirrors the LSTM levers)
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("policy_cls", ALL_POLICIES)
def test_separate_critic_core_is_built(policy_cls):
    policy = _make(policy_cls, shared_lstm=False, enable_critic_lstm=True)
    assert policy.lstm_critic is not None


@pytest.mark.parametrize("policy_cls", ALL_POLICIES)
def test_shared_core_has_no_critic_core(policy_cls):
    policy = _make(policy_cls, shared_lstm=True, enable_critic_lstm=False)
    assert policy.lstm_critic is None


@pytest.mark.parametrize("policy_cls", ALL_POLICIES)
def test_feedforward_critic_when_no_recurrent_core_for_critic(policy_cls):
    # shared_lstm=False AND enable_critic_lstm=False -> the critic is a plain Linear, not recurrent.
    policy = _make(policy_cls, shared_lstm=False, enable_critic_lstm=False)
    assert policy.lstm_critic is None
    assert isinstance(policy.critic, _nn.Linear)
    actions, values, _, _ = policy.forward(th.zeros(2, 32), _zero_states(policy, 2), th.zeros(2))
    assert values.shape == (2, 1)


def test_base_policy_make_core_is_abstract():
    with pytest.raises(NotImplementedError):
        RecurrentCoreActorCriticPolicy(OBS, ACT, lambda _: 1e-3, net_arch=[16], lstm_hidden_size=8)


def test_shared_gru_has_fewer_params_than_separate():
    separate = sum(p.numel() for p in _make(GRURecurrentActorCriticPolicy, shared_lstm=False, enable_critic_lstm=True).parameters())
    shared = sum(p.numel() for p in _make(GRURecurrentActorCriticPolicy, shared_lstm=True, enable_critic_lstm=False).parameters())
    assert shared < separate


def test_smaller_gru_hidden_size_has_fewer_params():
    big = sum(p.numel() for p in _make(GRURecurrentActorCriticPolicy, lstm_hidden_size=32).parameters())
    small = sum(p.numel() for p in _make(GRURecurrentActorCriticPolicy, lstm_hidden_size=4).parameters())
    assert small < big


# --------------------------------------------------------------------------------------
# forward() + predict() thread recurrent state with the right shapes
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("policy_cls", ALL_POLICIES)
def test_forward_returns_actions_values_and_updated_states(policy_cls):
    policy = _make(policy_cls)
    batch = 4
    obs = th.zeros(batch, 32)
    episode_starts = th.zeros(batch)
    actions, values, log_probs, states = policy.forward(obs, _zero_states(policy, batch), episode_starts)
    assert actions.shape == (batch,)
    assert values.shape == (batch, 1)
    assert log_probs.shape == (batch,)
    core = policy.lstm_actor
    assert states.pi[0].shape == (core.num_layers, batch, core.hidden_size)
    assert th.isfinite(values).all()


@pytest.mark.parametrize("policy_cls", ALL_POLICIES)
def test_predict_threads_state_from_none(policy_cls):
    policy = _make(policy_cls)
    obs = np.zeros((1, 32), dtype=np.float32)
    action, state = policy.predict(obs, state=None, episode_start=np.array([True]))
    assert action.shape == (1,)
    # State initialised from lstm_hidden_state_shape, then re-threadable.
    action2, state2 = policy.predict(obs, state=state, episode_start=np.array([False]))
    assert state2[0].shape == state[0].shape


# --------------------------------------------------------------------------------------
# End-to-end: RecurrentPPO actually trains with each core
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("policy_cls", ALL_POLICIES)
def test_recurrent_ppo_learns_a_few_steps(policy_cls):
    env = _TinyEnv()
    model = RecurrentPPO(
        policy_cls,
        env,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        device="cpu",
        policy_kwargs=dict(net_arch=[16], custom_net_arch=["Linear", "Linear"],
                           lstm_hidden_size=6, normalize_images=False),
    )
    model.learn(total_timesteps=16)
    action, _ = model.predict(env.observation_space.sample(), deterministic=True)
    assert action is not None
