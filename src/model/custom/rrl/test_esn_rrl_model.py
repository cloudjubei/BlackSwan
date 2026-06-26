"""Regression tests for ESN-RRL — the Echo State Network reservoir front-end (fixed, sub-unit spectral
radius, stateful, only the RRL readout trains). The policy/training/eval are RRL's, covered separately."""
import numpy as np
import torch

from src.conf.model_config import ModelConfig, ModelRLConfig
from src.model.custom.rrl.esn_rrl_model import ESNRRLModel, _RESERVOIR, _SPECTRAL_RADIUS
from src.model.custom.rrl.test_rrl_model import _Env  # reuse the fake env + data provider


def _model(env, episodes=2):
    rl = ModelRLConfig(
        model_name="esn-rrl", reward_model="combo_unified", net_arch=[64, 32], custom_net_arch=[],
        optimizer_class="Adam", activation_fn="ReLU", learning_rate=0.05, batch_size=16, gamma=0.99,
        seed=1, episodes=episodes,
    )
    return ESNRRLModel(ModelConfig(model_type="rl", model_rl=rl), env, "cpu")


def test_reservoir_has_echo_state_property_and_sets_feat_dim():
    m = _model(_Env(allow_short=False, steps=40, dim=4))
    assert m.feat_dim == _RESERVOIR
    assert m.proj.in_features == _RESERVOIR  # the trained readout reads the reservoir state
    spectral = torch.linalg.eigvals(m._w_res).abs().max().real.item()
    assert spectral <= _SPECTRAL_RADIUS + 1e-4  # sub-unit spectral radius = the echo-state property


def test_encode_step_is_stateful_and_resettable():
    m = _model(_Env(allow_short=False, steps=40, dim=4))
    x = np.ones(4, dtype=np.float32)
    s1 = m._encode_step(x).clone()
    s2 = m._encode_step(x)  # the leaky-integrator state is carried -> differs from the first step
    assert not torch.allclose(s1, s2)
    assert s1.shape == (_RESERVOIR,)
    m._reset_encoder()
    assert torch.allclose(m._encode_step(x), s1)  # after reset, the first-step state recurs


def test_reservoir_is_frozen_only_the_readout_trains():
    env = _Env(allow_short=False, steps=60, dim=4)
    m = _model(env)
    w_in0, w_res0, proj0 = m._w_in.clone(), m._w_res.clone(), m.proj.weight.clone()
    m.train(env)
    assert torch.equal(m._w_in, w_in0) and torch.equal(m._w_res, w_res0)  # reservoir stays frozen
    assert not torch.equal(m.proj.weight, proj0)  # only the readout learns
