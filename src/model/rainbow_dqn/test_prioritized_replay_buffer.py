"""Direct unit tests for the SB3-backed prioritized replay buffer.

This buffer subclasses ``stable_baselines3.common.buffers.ReplayBuffer`` and
adds proportional prioritization. We exercise ``add`` / ``sample`` /
``update_priorities`` on a tiny CPU buffer with a Box observation space and a
Discrete action space (no env, no GPU, no training loop). RNG-driven sampling
is seeded for determinism.

Several tests assert the INTENDED contract (priority bookkeeping that actually
stores per-transition priorities, importance-sampling weights, a non-degenerate
full-buffer sample) and are marked ``xfail`` because the current ``add`` writes
the priority to the wrong slot, leaving every priority at 0. See the module
docstring's bug notes in the agent report.
"""

import os
import sys

import numpy as np
import pytest
import torch as th
from gymnasium import spaces

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.model.rainbow_dqn.prioritized_replay_buffer import PrioritizedReplayBuffer


def _buffer(buffer_size=8, obs_dim=2, n_actions=3, **kw):
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Discrete(n_actions)
    return PrioritizedReplayBuffer(
        buffer_size=buffer_size,
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        **kw,
    )


def _add(buf, i, obs_dim=2, reward=1.0, done=0.0):
    obs = np.full((1, obs_dim), float(i), dtype=np.float32)
    next_obs = np.full((1, obs_dim), float(i + 1), dtype=np.float32)
    action = np.array([[i % buf.action_space.n]])
    buf.add(obs, next_obs, action, np.array([reward]), np.array([done]), [{}])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_init_sets_hyperparams_and_priority_array():
    buf = _buffer(buffer_size=8, alpha=0.7, beta=0.3, epsilon=1e-5)
    assert buf.alpha == pytest.approx(0.7)
    assert buf.beta == pytest.approx(0.3)
    assert buf.epsilon == pytest.approx(1e-5)
    assert buf.priorities.shape == (8,)
    assert np.all(buf.priorities == 0.0)


def test_default_hyperparams():
    buf = _buffer()
    assert buf.alpha == pytest.approx(0.6)
    assert buf.beta == pytest.approx(0.4)
    assert buf.epsilon == pytest.approx(1e-6)


# ---------------------------------------------------------------------------
# add: pointer / fullness bookkeeping (inherited behaviour we rely on)
# ---------------------------------------------------------------------------

def test_add_advances_pos_and_stores_observation():
    buf = _buffer(buffer_size=4)
    _add(buf, 0)
    assert buf.pos == 1
    assert not buf.full
    # the inherited storage records the observation at the slot just written.
    np.testing.assert_array_equal(buf.observations[0, 0], np.array([0.0, 0.0], dtype=np.float32))


def test_add_wraps_and_marks_full():
    buf = _buffer(buffer_size=4)
    for i in range(4):
        _add(buf, i)
    assert buf.full
    assert buf.pos == 0


# ---------------------------------------------------------------------------
# add: priority bookkeeping (BUG — see xfail)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="BUG: add() writes priority to self.pos AFTER super().add() "
                          "incremented it, and reads max from an all-zero array, so the "
                          "just-added transition never gets a priority.", strict=False)
def test_add_assigns_max_priority_to_new_transition():
    buf = _buffer(buffer_size=4)
    _add(buf, 0)
    # the very first transition should receive a positive (default 1.0) priority so it
    # can ever be sampled; intended PER behaviour. Currently it stays 0.0.
    assert buf.priorities[0] > 0.0


@pytest.mark.xfail(reason="BUG: priorities are written to the wrong slot, so after several "
                          "adds the stored priorities remain entirely zero.", strict=False)
def test_add_keeps_priorities_positive_after_several_adds():
    buf = _buffer(buffer_size=4)
    for i in range(3):
        _add(buf, i)
    # each stored transition should have a positive priority.
    assert np.all(buf.priorities[:3] > 0.0)


def test_add_currently_leaves_priorities_zero_characterization():
    # Characterization of the present (buggy) behaviour: priorities never become
    # non-zero through add() alone.
    buf = _buffer(buffer_size=4)
    for i in range(3):
        _add(buf, i)
    assert np.all(buf.priorities == 0.0)


# ---------------------------------------------------------------------------
# update_priorities
# ---------------------------------------------------------------------------

def test_update_priorities_writes_with_epsilon_offset():
    buf = _buffer(buffer_size=8, epsilon=1e-6)
    for i in range(4):
        _add(buf, i)
    buf.update_priorities(np.array([0, 1, 2, 3]), np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(
        buf.priorities[:4],
        np.array([1.0, 2.0, 3.0, 4.0]) + 1e-6,
        rtol=1e-6,
    )
    # untouched slots stay at zero.
    assert np.all(buf.priorities[4:] == 0.0)


def test_update_priorities_single_index():
    buf = _buffer(buffer_size=8)
    for i in range(3):
        _add(buf, i)
    buf.update_priorities(np.array([1]), np.array([5.0], dtype=np.float32))
    assert buf.priorities[1] == pytest.approx(5.0 + buf.epsilon)
    assert buf.priorities[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# sample: non-full path (works because of the +1e-6 guard)
# ---------------------------------------------------------------------------

def test_sample_not_full_returns_valid_indices_and_weights():
    buf = _buffer(buffer_size=8)
    for i in range(4):
        _add(buf, i)
    np.random.seed(0)
    samples, weights, indices = buf.sample(batch_size=3)
    # only the 4 filled slots may be sampled while not full.
    assert all(0 <= i < 4 for i in indices)
    assert len(indices) == 3
    assert isinstance(weights, th.Tensor)
    assert weights.shape == (3,)
    # uniform (all-zero) priorities -> normalized weights are all 1.
    np.testing.assert_allclose(weights.cpu().numpy(), np.ones(3), rtol=1e-5)


def test_sample_returns_replaybuffersamples_shape():
    buf = _buffer(buffer_size=8, obs_dim=2)
    for i in range(5):
        _add(buf, i)
    np.random.seed(1)
    samples, weights, indices = buf.sample(batch_size=4)
    # SB3 ReplayBufferSamples carries observations matching the requested batch.
    assert samples.observations.shape[0] == 4
    assert samples.observations.shape[1] == 2


def test_sample_weights_normalized_to_unit_max_after_update():
    buf = _buffer(buffer_size=8, alpha=1.0, beta=0.5)
    for i in range(4):
        _add(buf, i)
    buf.update_priorities(np.array([0, 1, 2, 3]), np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    np.random.seed(2)
    samples, weights, indices = buf.sample(batch_size=4)
    w = weights.cpu().numpy()
    # weights are divided by their own max -> max is 1.0 and all are in (0, 1].
    assert w.max() == pytest.approx(1.0)
    assert np.all(w > 0.0)
    assert np.all(w <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# sample: full path (BUG — all-zero priorities divide by zero -> NaN)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="BUG: when full, sample() uses raw priorities without the +1e-6 "
                          "guard; combined with the add() bug they are all zero, so "
                          "probabilities /= 0 yields NaN and np.random.choice raises.",
                   strict=False, raises=(ValueError, ZeroDivisionError, FloatingPointError))
def test_sample_full_buffer_does_not_produce_nan():
    buf = _buffer(buffer_size=4)
    for i in range(4):  # fills the buffer
        _add(buf, i)
    assert buf.full
    np.random.seed(0)
    # intended: a full buffer is samplable. Currently this raises on NaN probabilities.
    samples, weights, indices = buf.sample(batch_size=2)
    assert all(0 <= i < 4 for i in indices)
    assert not np.any(np.isnan(weights.cpu().numpy()))


def test_sample_full_buffer_after_priority_update_recovers():
    # If priorities are explicitly set (the intended invariant), the full-buffer path
    # works — proving the failure above is specifically the all-zero priority bug.
    buf = _buffer(buffer_size=4, alpha=1.0, beta=0.5)
    for i in range(4):
        _add(buf, i)
    assert buf.full
    buf.update_priorities(np.array([0, 1, 2, 3]), np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    np.random.seed(0)
    samples, weights, indices = buf.sample(batch_size=3)
    assert all(0 <= i < 4 for i in indices)
    w = weights.cpu().numpy()
    assert not np.any(np.isnan(w))
    assert w.max() == pytest.approx(1.0)
