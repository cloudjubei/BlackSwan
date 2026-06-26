"""Regression tests for EnsembleModel — the discrete-action ensemble vote (average member Q-values, then
argmax) and the duck-typed BaseAlgorithm surface RLModel drives. Hermetic: fake members, no env/torch DQN.
(Before the fix EnsembleModel subclassed OffPolicyAlgorithm without a real super().__init__, so it had no
`device` and inherited a broken `learn`.)"""
import numpy as np
import pytest
import torch as th

from src.model.custom.ensemble.ensemble import EnsembleModel


class _FakeMember:
    """A minimal stand-in for a trained DQN member: a fixed Q-row, plus the policy/q_net/learn/save/logger
    surface EnsembleModel touches."""

    def __init__(self, q_row):
        self._q = th.tensor([q_row], dtype=th.float32)
        self.device = "cpu"
        self.policy = self  # policy.obs_to_tensor / set_training_mode live here in this fake
        self.learned = 0
        self.saved = None
        self.logger = None
        self.training_mode = None

    def obs_to_tensor(self, obs):
        return th.as_tensor(obs, dtype=th.float32), None

    def set_training_mode(self, mode):
        self.training_mode = mode

    def q_net(self, obs_tensor):
        return self._q

    def learn(self, total_timesteps, **kwargs):
        self.learned += total_timesteps

    def save(self, path):
        self.saved = path

    def set_logger(self, logger):
        self.logger = logger


def test_predict_averages_member_q_values_then_argmax():
    # A prefers action 1 (Q=[1,5,2]); B prefers action 0 (Q=[4,1,2]). mean=[2.5,3,2] -> argmax=1.
    a, b = _FakeMember([1.0, 5.0, 2.0]), _FakeMember([4.0, 1.0, 2.0])
    action, state = EnsembleModel([a, b]).predict(np.zeros((1, 3), dtype=np.float32), state="S")
    assert action.tolist() == [1]
    assert state == "S"  # non-recurrent: state passes straight through
    assert a.training_mode is False and b.training_mode is False  # members put in eval mode


def test_predict_returns_numpy_action_array_and_state_tuple():
    action, state = EnsembleModel([_FakeMember([0.0, 9.0])]).predict(np.zeros((1, 2), dtype=np.float32))
    assert isinstance(action, np.ndarray) and action.tolist() == [1] and state is None


def test_learn_trains_every_member_with_the_same_budget():
    a, b = _FakeMember([1.0, 0.0]), _FakeMember([0.0, 1.0])
    EnsembleModel([a, b]).learn(total_timesteps=100, progress_bar=False, reset_num_timesteps=True)
    assert a.learned == 100 and b.learned == 100


def test_save_logger_and_device_delegate_to_members():
    a, b = _FakeMember([1.0]), _FakeMember([2.0])
    ens = EnsembleModel([a, b])
    assert ens.device == "cpu" and ens.policy is a.policy  # exposed for RLModel's eval-fold + device reads
    ens.set_logger("LOG")
    ens.save("/tmp/ck")
    assert a.logger == "LOG" and b.logger == "LOG"
    assert a.saved == "/tmp/ck_member0" and b.saved == "/tmp/ck_member1"  # one checkpoint per member


def test_empty_ensemble_is_rejected():
    with pytest.raises(ValueError):
        EnsembleModel([])
