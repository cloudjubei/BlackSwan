"""Direct unit tests for the SB3-backed prioritized replay buffer.

This buffer subclasses ``stable_baselines3.common.buffers.ReplayBuffer`` and
adds proportional prioritization. We exercise ``add`` / ``sample`` /
``update_priorities`` on a tiny CPU buffer with a Box observation space and a
Discrete action space (no env, no GPU, no training loop). RNG-driven sampling
is seeded for determinism.

These tests assert the INTENDED contract: ``add`` seeds each new transition with
a positive max priority in the correct slot, ``update_priorities`` applies the
epsilon offset, and a full buffer is samplable without NaN probabilities (the
epsilon floor binds to both the full and not-full branches of ``sample``).
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

def test_add_assigns_max_priority_to_new_transition():
    buf = _buffer(buffer_size=4)
    _add(buf, 0)
    # the very first transition receives a positive (default 1.0) priority, seeded in
    # the slot written by super().add() (captured BEFORE pos is incremented).
    assert buf.priorities[0] == pytest.approx(1.0)


def test_add_keeps_priorities_positive_after_several_adds():
    buf = _buffer(buffer_size=4)
    for i in range(3):
        _add(buf, i)
    # each stored transition lands its priority in the correct slot and stays positive.
    assert np.all(buf.priorities[:3] > 0.0)
    # the not-yet-written slot remains zero (priority went to the right place).
    assert buf.priorities[3] == pytest.approx(0.0)


def test_add_seeds_new_transition_with_current_max_priority():
    # After raising one slot's priority, a subsequent add() seeds the new transition
    # with the current maximum (not the dead all-zero default).
    buf = _buffer(buffer_size=4)
    _add(buf, 0)
    buf.update_priorities(np.array([0]), np.array([5.0], dtype=np.float32))
    _add(buf, 1)
    # the freshly added slot 1 inherits the running max priority.
    assert buf.priorities[1] == pytest.approx(buf.priorities[:2].max())
    assert buf.priorities[1] >= 5.0


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
    # slot 0 was not updated, so it keeps the default priority seeded by add().
    assert buf.priorities[0] == pytest.approx(1.0)


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
    # all four transitions seeded to the same default (1.0) priority -> uniform
    # probabilities -> normalized weights are all 1.
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

def test_sample_full_buffer_does_not_produce_nan():
    buf = _buffer(buffer_size=4)
    for i in range(4):  # fills the buffer
        _add(buf, i)
    assert buf.full
    np.random.seed(0)
    # a full buffer is samplable: the epsilon floor now binds to the full branch too,
    # so no zero priorities -> no NaN probabilities.
    samples, weights, indices = buf.sample(batch_size=2)
    assert all(0 <= i < 4 for i in indices)
    assert len(indices) == 2
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


def test_sample_full_buffer_with_zero_priorities_uses_epsilon_floor():
    # Guards bug B *independently* of the add()-seeding fix: force a FULL buffer that contains only
    # zero priorities (the state the seeding fix would normally prevent) and assert the full-branch
    # epsilon floor keeps probabilities finite. Without the floor on the full branch, 0**alpha = 0
    # -> divide-by-zero -> NaN -> np.random.choice raises ValueError.
    buf = _buffer(buffer_size=4)
    for i in range(4):  # fill -> full
        _add(buf, i)
    assert buf.full
    buf.priorities[:] = 0.0  # pathological: a full buffer with no positive priority
    np.random.seed(0)
    samples, weights, indices = buf.sample(batch_size=2)
    assert all(0 <= i < 4 for i in indices)
    assert not np.any(np.isnan(weights.cpu().numpy()))


def test_sample_floor_uses_configured_epsilon_not_hardcoded():
    # Guards bug C: the sampling floor must be the configurable self.epsilon, not a hardcoded 1e-6.
    # With a large epsilon a zero-priority slot still draws meaningful probability; with the old
    # 1e-6 literal it would be sampled essentially never.
    buf = _buffer(buffer_size=8, alpha=1.0, epsilon=0.5)
    _add(buf, 0)
    _add(buf, 1)  # not full -> sample() uses self.priorities[:pos] (the branch that held the literal)
    buf.priorities[0] = 1.0
    buf.priorities[1] = 0.0
    # alpha=1, epsilon=0.5 -> probabilities ∝ [1.5, 0.5] -> p(slot 1) = 0.25; the old 1e-6 literal
    # would give p(slot 1) ≈ 1e-6 ≈ 0.
    np.random.seed(0)
    n, hits = 4000, 0
    for _ in range(n):
        _, _, idx = buf.sample(batch_size=1)
        hits += int(idx[0] == 1)
    assert hits / n > 0.12
