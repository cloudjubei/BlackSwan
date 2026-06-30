"""Direct tests for RLModel (the stable-baselines3 wrapper) in rl_model.py.

RLModel wraps a sb3 ``BaseAlgorithm``. We bypass __init__ with ``__new__`` and inject a tiny fake
sb3 model (with ``learn`` / ``save`` / ``predict``) plus a fake env, so the predict/test/train
control flow and the action remapping are exercised with no real training, GPU, or sb3 policy.
``progress_bar=False`` is used in test() to skip the heavy DQN/ProgressBar instantiation.
"""

import os

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize

from src.conf.model_config import ModelConfig, ModelRLConfig
from src.model.rl_model import RLModel, is_recurrent_model_name, fold_eval_parametrizations


# --- fakes -----------------------------------------------------------------

class _FakeSB3:
    """Stands in for a stable-baselines3 algorithm: records learn/save calls, returns a fixed action."""

    def __init__(self, action=1, reppo_state="lstm"):
        self._action = action
        self._reppo_state = reppo_state
        self.learn_calls = []
        self.saved_path = None
        self.predict_calls = []

    def learn(self, total_timesteps, progress_bar, log_interval, reset_num_timesteps):
        self.learn_calls.append(
            dict(total_timesteps=total_timesteps, reset_num_timesteps=reset_num_timesteps)
        )

    def save(self, path):
        self.saved_path = path

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        # supports both the plain (obs, deterministic) and the reppo (obs, state, episode_start) call.
        self.predict_calls.append(
            dict(obs=obs, state=state, episode_start=episode_start, deterministic=deterministic)
        )
        return (np.array(self._action), self._reppo_state)


class _FakeEnv:
    def __init__(self, n_steps, done_action_marker=None):
        self.n_steps = n_steps
        self.i = 0
        self.actions = []
        self.last_obs = np.zeros(3, dtype=np.float32)

    def reset(self):
        self.i = 0
        return (np.zeros(3, dtype=np.float32), {})

    def step(self, action):
        self.actions.append(action)
        self.i += 1
        done = self.i >= self.n_steps
        return (np.zeros(3, dtype=np.float32), 0.0, done, False, {})

    def get_timesteps(self):
        return self.n_steps


def _rl_model(sb3, **rl_kw):
    defaults = dict(model_name="dqn", reward_model="combo_all", net_arch=[], custom_net_arch=[""])
    defaults.update(rl_kw)
    cfg = ModelConfig(model_type="rl", model_rl=ModelRLConfig(**defaults))
    m = RLModel.__new__(RLModel)
    m.config = cfg
    m.rl_config = cfg.model_rl
    m.rl_model = sb3
    m.id = "rl-test-id"
    return m


# --- train -----------------------------------------------------------------

def test_train_learns_once_per_episode_and_resets_only_first():
    sb3 = _FakeSB3()
    m = _rl_model(sb3, episodes=3, progress_bar=False, checkpoints_folder="cp")
    m.train(_FakeEnv(7))
    assert len(sb3.learn_calls) == 3
    # only the first episode resets the step counter; the rest continue it.
    assert [c["reset_num_timesteps"] for c in sb3.learn_calls] == [True, False, False]
    # timesteps come from the env.
    assert all(c["total_timesteps"] == 7 for c in sb3.learn_calls)


def test_train_saves_to_checkpoints_folder_joined_with_id(tmp_path):
    sb3 = _FakeSB3()
    m = _rl_model(sb3, episodes=1, progress_bar=False, checkpoints_folder=str(tmp_path))
    m.train(_FakeEnv(5))
    assert sb3.saved_path == os.path.join(str(tmp_path), "rl-test-id")


# --- test() control flow ---------------------------------------------------

def test_test_non_reppo_steps_until_done():
    sb3 = _FakeSB3(action=1)
    m = _rl_model(sb3, model_name="dqn")
    env = _FakeEnv(4)
    m.test(env, deterministic=True, progress_bar=False)
    assert len(env.actions) == 4
    # plain predict path: state/episode_start are never passed.
    assert all(c["state"] is None for c in sb3.predict_calls)


def test_test_passes_deterministic_flag_through():
    sb3 = _FakeSB3(action=1)
    m = _rl_model(sb3, model_name="dqn")
    m.test(_FakeEnv(2), deterministic=False, progress_bar=False)
    assert all(c["deterministic"] is False for c in sb3.predict_calls)


def test_test_reppo_branch_uses_recurrent_predict():
    # the 'reppo' model name takes the recurrent branch that threads lstm state + episode_start.
    sb3 = _FakeSB3(action=1)
    m = _rl_model(sb3, model_name="reppo")
    env = _FakeEnv(3)
    m.test(env, deterministic=True, progress_bar=False)
    assert len(env.actions) == 3
    # first call starts with no state; subsequent calls receive the returned lstm state.
    assert sb3.predict_calls[0]["state"] is None
    assert sb3.predict_calls[1]["state"] == "lstm"
    # episode_start begins as a True-filled array.
    assert bool(np.all(sb3.predict_calls[0]["episode_start"]))


def test_test_reppo_custom_branch_uses_recurrent_predict():
    # 'reppo-custom' is ALSO RecurrentPPO (model_factory) — it must thread lstm state at eval, not run
    # memoryless. Regression for the bug where only the bare 'reppo' name took the recurrent branch.
    sb3 = _FakeSB3(action=1)
    m = _rl_model(sb3, model_name="reppo-custom")
    env = _FakeEnv(3)
    m.test(env, deterministic=True, progress_bar=False)
    assert len(env.actions) == 3
    assert sb3.predict_calls[0]["state"] is None
    assert sb3.predict_calls[1]["state"] == "lstm"
    assert bool(np.all(sb3.predict_calls[0]["episode_start"]))


def test_is_recurrent_model_name_covers_all_recurrent_cores():
    # LSTM, GRU and S4D cores all thread recurrent state at eval and must take the recurrent branch.
    for name in ("reppo", "reppo-custom", "gru", "gru-custom", "s4d", "s4d-custom"):
        assert is_recurrent_model_name(name), name
    assert not is_recurrent_model_name("dqn")
    assert not is_recurrent_model_name(None)


# --- A7: eval weight-norm fold ---------------------------------------------

def test_fold_eval_parametrizations_is_numerically_identical():
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.utils.parametrizations.weight_norm(nn.Linear(8, 4)), nn.ReLU(), nn.Linear(4, 2)
    )
    net.eval()
    x = torch.randn(3, 8)
    before = net(x).detach().clone()
    assert parametrize.is_parametrized(net[0], "weight")
    folded = fold_eval_parametrizations(net)
    assert folded == 1
    assert not parametrize.is_parametrized(net[0], "weight")
    assert torch.allclose(before, net(x).detach(), atol=1e-6)


def test_fold_eval_parametrizations_is_noop_without_reparam():
    assert fold_eval_parametrizations(nn.Linear(4, 2)) == 0


def test_fold_eval_parametrizations_is_idempotent():
    net = nn.Sequential(nn.utils.parametrizations.weight_norm(nn.Linear(4, 2)))
    assert fold_eval_parametrizations(net) == 1
    assert fold_eval_parametrizations(net) == 0


def test_fold_eliminates_the_weight_norm_recompute(monkeypatch):
    # SPEEDUP PROOF (A7): pre-fold, EVERY forward recomputes the weight from (g, v) via torch._weight_norm
    # (the op cProfile flagged at ~8% of the reppo-custom forward). Folding must make that op stop running
    # — proven by counting calls to torch._weight_norm before vs after the fold.
    net = nn.Sequential(
        nn.utils.parametrizations.weight_norm(nn.Linear(16, 16)),
        nn.ReLU(),
        nn.utils.parametrizations.weight_norm(nn.Linear(16, 4)),
    )
    net.eval()
    x = torch.randn(2, 16)
    count = {"n": 0}
    real = torch._weight_norm

    def counting(*args, **kwargs):
        count["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "_weight_norm", counting)
    net(x)
    assert count["n"] >= 2  # both weight_norm layers recompute on a pre-fold forward
    fold_eval_parametrizations(net)
    count["n"] = 0
    net(x)
    assert count["n"] == 0  # post-fold: the per-forward weight_norm recompute is gone


def test_test_folds_policy_parametrizations():
    # test() bakes weight_norm into plain weights for the eval passes; best-effort, so a policy that
    # isn't a torch module (sbx/JAX) or missing entirely just no-ops (covered by the other test() tests).
    sb3 = _FakeSB3(action=1)
    sb3.policy = nn.Sequential(nn.utils.parametrizations.weight_norm(nn.Linear(3, 2)))
    m = _rl_model(sb3, model_name="dqn")
    m.test(_FakeEnv(2), deterministic=True, progress_bar=False)
    assert not parametrize.is_parametrized(sb3.policy[0], "weight")


# --- predict / predictOnline action remapping ------------------------------

def test_predict_maps_action_2_to_minus_one():
    sb3 = _FakeSB3(action=2)
    m = _rl_model(sb3)
    env = _FakeEnv(1)
    assert m.predict(env, deterministic=True) == -1
    # predict steps the env once with the raw action.
    assert len(env.actions) == 1


def test_predict_passes_through_non_two_actions():
    for raw, expected in [(0, 0), (1, 1)]:
        sb3 = _FakeSB3(action=raw)
        m = _rl_model(sb3)
        assert m.predict(_FakeEnv(1)) == expected


def test_predict_uses_last_obs_not_reset():
    sb3 = _FakeSB3(action=1)
    m = _rl_model(sb3)
    env = _FakeEnv(1)
    sentinel = np.array([9.0, 9.0, 9.0], dtype=np.float32)
    env.last_obs = sentinel
    m.predict(env)
    # the observation handed to predict is env.last_obs (online inference, no reset).
    assert np.array_equal(sb3.predict_calls[0]["obs"], sentinel)
    assert len(env.actions) == 1


def test_predict_online_maps_and_unwraps_ndarray():
    sb3 = _FakeSB3(action=2)
    m = _rl_model(sb3)
    obs = np.zeros(3, dtype=np.float32)
    # ndarray action 2 -> .item() -> 2 -> remapped to -1, without stepping any env.
    assert m.predictOnline(obs, deterministic=True) == -1


def test_predict_online_passes_through_scalar_action():
    sb3 = _FakeSB3(action=0)
    m = _rl_model(sb3)
    assert m.predictOnline(np.zeros(3, dtype=np.float32)) == 0
