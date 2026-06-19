"""Direct unit tests for the rainbow numpy replay buffers.

Covers ``ReplayBuffer`` (store / sample / n-step discounting / wrap-on-full)
and ``PrioritizedReplayBuffer`` (priority bookkeeping, proportional sampling,
importance-sampling weights). Pure numpy + the local segment trees: no torch,
no env, no GPU. Sampling that uses RNG is made deterministic with explicit
seeds. A repo-root sys.path insert mirrors ``test_base_crypto_env.py``.
"""

import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.model.rainbow.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


def _obs(v, dim=2):
    return np.full(dim, float(v), dtype=np.float32)


# ---------------------------------------------------------------------------
# ReplayBuffer: store / len / wrap
# ---------------------------------------------------------------------------

def test_store_single_step_returns_transition_and_advances():
    buf = ReplayBuffer(obs_dim=2, size=4, batch_size=2, n_step=1)
    out = buf.store(_obs(1), act=0.0, rew=1.5, next_obs=_obs(2), done=False)
    # n_step == 1 -> the just-appended transition is returned immediately.
    assert out != ()
    assert len(buf) == 1
    assert buf.ptr == 1
    np.testing.assert_array_equal(buf.obs_buf[0], _obs(1))
    np.testing.assert_array_equal(buf.next_obs_buf[0], _obs(2))
    assert buf.rews_buf[0] == pytest.approx(1.5)
    assert buf.acts_buf[0] == pytest.approx(0.0)
    assert buf.done_buf[0] == pytest.approx(0.0)


def test_len_starts_at_zero():
    buf = ReplayBuffer(obs_dim=1, size=4)
    assert len(buf) == 0


def test_store_wraps_pointer_and_caps_size():
    buf = ReplayBuffer(obs_dim=1, size=3, batch_size=1, n_step=1)
    for i in range(5):
        buf.store(_obs(i, 1), act=float(i), rew=float(i), next_obs=_obs(i + 1, 1), done=False)
    # size caps at max_size; ptr wraps modulo max_size (5 % 3 == 2).
    assert len(buf) == 3
    assert buf.ptr == 2
    # the oldest two slots were overwritten by the later stores (3 and 4).
    assert buf.acts_buf[0] == pytest.approx(3.0)
    assert buf.acts_buf[1] == pytest.approx(4.0)
    assert buf.acts_buf[2] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# ReplayBuffer: n-step learning
# ---------------------------------------------------------------------------

def test_store_n_step_not_ready_returns_empty_tuple():
    buf = ReplayBuffer(obs_dim=1, size=4, n_step=3)
    # first two stores cannot form a 3-step transition yet.
    assert buf.store(_obs(0, 1), 0.0, 1.0, _obs(1, 1), False) == ()
    assert buf.store(_obs(1, 1), 0.0, 1.0, _obs(2, 1), False) == ()
    assert len(buf) == 0
    # third store completes the window.
    out = buf.store(_obs(2, 1), 0.0, 1.0, _obs(3, 1), False)
    assert out != ()
    assert len(buf) == 1


def test_get_n_step_info_discounts_rewards():
    buf = ReplayBuffer(obs_dim=1, size=4, n_step=3, gamma=0.5)
    buf.store(_obs(0, 1), 0.0, 1.0, _obs(1, 1), False)
    buf.store(_obs(1, 1), 0.0, 2.0, _obs(2, 1), False)
    buf.store(_obs(2, 1), 0.0, 4.0, _obs(3, 1), False)
    # rew = 1 + 0.5*(2 + 0.5*4) = 1 + 0.5*4 = 3.0 ; obs is the FIRST transition obs.
    assert buf.rews_buf[0] == pytest.approx(3.0)
    np.testing.assert_array_equal(buf.obs_buf[0], _obs(0, 1))
    # next_obs / done come from the last (non-terminal) transition.
    np.testing.assert_array_equal(buf.next_obs_buf[0], _obs(3, 1))
    assert buf.done_buf[0] == pytest.approx(0.0)


def test_get_n_step_info_stops_at_done():
    buf = ReplayBuffer(obs_dim=1, size=4, n_step=3, gamma=0.9)
    buf.store(_obs(0, 1), 0.0, 1.0, _obs(1, 1), False)
    # the middle transition is terminal -> later reward is masked out.
    buf.store(_obs(1, 1), 0.0, 5.0, _obs(2, 1), True)
    buf.store(_obs(2, 1), 0.0, 9.0, _obs(3, 1), False)
    # walking backwards from the last: rew=9; then middle d=True -> rew = 5 + 0.9*9*(1-1)=5,
    # next_obs/done snap to the terminal middle transition (obs 2, done True);
    # then first d=False -> rew = 1 + 0.9*5 = 5.5.
    assert buf.rews_buf[0] == pytest.approx(5.5)
    np.testing.assert_array_equal(buf.next_obs_buf[0], _obs(2, 1))
    assert buf.done_buf[0] == pytest.approx(1.0)


def test_get_n_step_info_direct_single_transition():
    buf = ReplayBuffer.__new__(ReplayBuffer)
    from collections import deque
    dq = deque([(_obs(0, 1), 0.0, 7.0, _obs(1, 1), True)])
    rew, next_obs, done = buf._get_n_step_info(dq, gamma=0.9)
    assert rew == pytest.approx(7.0)
    np.testing.assert_array_equal(next_obs, _obs(1, 1))
    assert done is True


# ---------------------------------------------------------------------------
# ReplayBuffer: sampling
# ---------------------------------------------------------------------------

def test_sample_batch_shapes_and_indices():
    buf = ReplayBuffer(obs_dim=2, size=10, batch_size=4, n_step=1)
    for i in range(10):
        buf.store(_obs(i), float(i), float(i), _obs(i + 1), False)
    np.random.seed(0)
    batch = buf.sample_batch()
    assert set(batch.keys()) == {"obs", "next_obs", "acts", "rews", "done", "indices"}
    assert batch["obs"].shape == (4, 2)
    assert batch["indices"].shape == (4,)
    # replace=False -> all sampled indices distinct and within size.
    assert len(set(batch["indices"].tolist())) == 4
    assert all(0 <= i < 10 for i in batch["indices"])


def test_sample_batch_from_idxs_selects_requested_rows():
    buf = ReplayBuffer(obs_dim=2, size=10, batch_size=4, n_step=1)
    for i in range(10):
        buf.store(_obs(i), act=float(i * 10), rew=float(i), next_obs=_obs(i + 1), done=False)
    idxs = np.array([2, 5, 7])
    out = buf.sample_batch_from_idxs(idxs)
    np.testing.assert_array_equal(out["acts"], np.array([20.0, 50.0, 70.0], dtype=np.float32))
    assert out["obs"].shape == (3, 2)
    assert "indices" not in out  # this variant intentionally omits indices.


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer
# ---------------------------------------------------------------------------

def test_prioritized_init_rejects_negative_alpha():
    with pytest.raises(AssertionError):
        PrioritizedReplayBuffer(obs_dim=1, size=4, alpha=-0.1)


def test_prioritized_tree_capacity_is_next_power_of_two():
    buf = PrioritizedReplayBuffer(obs_dim=1, size=5, batch_size=2)
    # next power of two >= 5 is 8.
    assert buf.sum_tree.capacity == 8
    assert buf.min_tree.capacity == 8


def test_prioritized_tree_capacity_exact_power_of_two():
    buf = PrioritizedReplayBuffer(obs_dim=1, size=8, batch_size=2)
    assert buf.sum_tree.capacity == 8


def test_prioritized_store_sets_max_priority_in_trees():
    buf = PrioritizedReplayBuffer(obs_dim=1, size=4, batch_size=2, alpha=0.6, n_step=1)
    buf.store(_obs(0, 1), 0, 1.0, _obs(1, 1), False)
    # default max_priority is 1.0 ; 1.0 ** alpha == 1.0 written into both trees.
    assert buf.sum_tree[0] == pytest.approx(1.0)
    assert buf.min_tree[0] == pytest.approx(1.0)
    assert buf.tree_ptr == 1


def test_prioritized_store_nstep_not_ready_does_not_touch_trees():
    buf = PrioritizedReplayBuffer(obs_dim=1, size=4, batch_size=2, n_step=2)
    out = buf.store(_obs(0, 1), 0, 1.0, _obs(1, 1), False)
    assert out == ()
    # no n-step transition yet -> tree pointer untouched.
    assert buf.tree_ptr == 0
    assert buf.sum_tree.sum() == pytest.approx(0.0)


def test_prioritized_update_priorities_writes_alpha_weighted_and_tracks_max():
    buf = PrioritizedReplayBuffer(obs_dim=1, size=4, batch_size=2, alpha=0.5, n_step=1)
    for i in range(3):
        buf.store(_obs(i, 1), 0, 1.0, _obs(i + 1, 1), False)
    buf.update_priorities([0, 1], np.array([4.0, 9.0]))
    # priority ** alpha with alpha=0.5 -> sqrt.
    assert buf.sum_tree[0] == pytest.approx(2.0)
    assert buf.sum_tree[1] == pytest.approx(3.0)
    assert buf.min_tree[0] == pytest.approx(2.0)
    # max_priority tracks the largest seen priority.
    assert buf.max_priority == pytest.approx(9.0)


def test_prioritized_update_priorities_rejects_mismatched_lengths():
    buf = PrioritizedReplayBuffer(obs_dim=1, size=4, n_step=1)
    buf.store(_obs(0, 1), 0, 1.0, _obs(1, 1), False)
    with pytest.raises(AssertionError):
        buf.update_priorities([0], np.array([1.0, 2.0]))


def test_prioritized_update_priorities_rejects_nonpositive_priority():
    buf = PrioritizedReplayBuffer(obs_dim=1, size=4, n_step=1)
    buf.store(_obs(0, 1), 0, 1.0, _obs(1, 1), False)
    with pytest.raises(AssertionError):
        buf.update_priorities([0], np.array([0.0]))


def test_prioritized_update_priorities_rejects_out_of_range_index():
    buf = PrioritizedReplayBuffer(obs_dim=1, size=4, n_step=1)
    buf.store(_obs(0, 1), 0, 1.0, _obs(1, 1), False)  # len == 1
    with pytest.raises(AssertionError):
        buf.update_priorities([5], np.array([1.0]))


def test_calculate_weight_uniform_priorities_is_one():
    # equal priorities => every importance-sampling weight equals max_weight => 1.0.
    buf = PrioritizedReplayBuffer(obs_dim=1, size=4, batch_size=2, alpha=0.6, n_step=1)
    for i in range(4):
        buf.store(_obs(i, 1), 0, 1.0, _obs(i + 1, 1), False)
    for i in range(4):
        assert buf._calculate_weight(i, beta=0.4) == pytest.approx(1.0)


def test_calculate_weight_lowest_priority_has_unit_weight():
    # the min-priority sample defines max_weight, so its normalized weight is 1.0;
    # higher-priority samples get weights < 1.
    buf = PrioritizedReplayBuffer(obs_dim=1, size=4, batch_size=2, alpha=1.0, n_step=1)
    for i in range(3):
        buf.store(_obs(i, 1), 0, 1.0, _obs(i + 1, 1), False)
    buf.update_priorities([0, 1, 2], np.array([1.0, 2.0, 4.0]))
    w_min = buf._calculate_weight(0, beta=0.5)  # idx 0 has the smallest priority
    w_high = buf._calculate_weight(2, beta=0.5)
    assert w_min == pytest.approx(1.0)
    assert w_high < 1.0


def test_calculate_weight_matches_closed_form():
    buf = PrioritizedReplayBuffer(obs_dim=1, size=4, batch_size=2, alpha=1.0, n_step=1)
    for i in range(3):
        buf.store(_obs(i, 1), 0, 1.0, _obs(i + 1, 1), False)
    buf.update_priorities([0, 1, 2], np.array([1.0, 2.0, 4.0]))
    beta = 0.5
    n = len(buf)
    total = buf.sum_tree.sum()
    p_min = buf.min_tree.min() / total
    max_weight = (p_min * n) ** (-beta)
    idx = 2
    p_sample = buf.sum_tree[idx] / total
    expected = ((p_sample * n) ** (-beta)) / max_weight
    assert buf._calculate_weight(idx, beta) == pytest.approx(expected)


def test_sample_proportional_only_returns_valid_indices():
    buf = PrioritizedReplayBuffer(obs_dim=1, size=8, batch_size=4, alpha=1.0, n_step=1)
    # store 5 transitions; only indices 0..4 are valid.
    for i in range(5):
        buf.store(_obs(i, 1), 0, 1.0, _obs(i + 1, 1), False)
    random.seed(7)
    idxs = buf._sample_proportional()
    assert len(idxs) == 4
    assert all(0 <= i < len(buf) for i in idxs)


def test_sample_proportional_avoids_zero_priority_indices():
    # give index 0 a huge priority and others tiny; sampling should heavily favor 0
    # and never return an index beyond len(self).
    buf = PrioritizedReplayBuffer(obs_dim=1, size=8, batch_size=8, alpha=1.0, n_step=1)
    for i in range(4):
        buf.store(_obs(i, 1), 0, 1.0, _obs(i + 1, 1), False)
    buf.update_priorities([0, 1, 2, 3], np.array([1000.0, 1e-6, 1e-6, 1e-6]))
    random.seed(0)
    idxs = buf._sample_proportional()
    assert all(0 <= i < 4 for i in idxs)
    assert idxs.count(0) >= 4  # dominant-priority index is selected most of the time.


def test_sample_batch_requires_enough_samples():
    buf = PrioritizedReplayBuffer(obs_dim=1, size=8, batch_size=4, n_step=1)
    for i in range(2):
        buf.store(_obs(i, 1), 0, 1.0, _obs(i + 1, 1), False)
    with pytest.raises(AssertionError):
        buf.sample_batch(beta=0.4)


def test_sample_batch_rejects_nonpositive_beta():
    buf = PrioritizedReplayBuffer(obs_dim=1, size=8, batch_size=2, n_step=1)
    for i in range(4):
        buf.store(_obs(i, 1), 0, 1.0, _obs(i + 1, 1), False)
    with pytest.raises(AssertionError):
        buf.sample_batch(beta=0.0)


def test_sample_batch_returns_weights_and_indices():
    buf = PrioritizedReplayBuffer(obs_dim=2, size=8, batch_size=4, alpha=0.6, n_step=1)
    for i in range(6):
        buf.store(_obs(i), 0, float(i), _obs(i + 1), False)
    random.seed(3)
    np.random.seed(3)
    batch = buf.sample_batch(beta=0.4)
    assert set(batch.keys()) == {
        "obs", "next_obs", "acts", "rews", "done", "weights", "indices",
    }
    assert batch["obs"].shape == (4, 2)
    assert batch["weights"].shape == (4,)
    assert len(batch["indices"]) == 4
    # equal priorities so far -> all weights == 1.0.
    np.testing.assert_allclose(batch["weights"], np.ones(4), rtol=1e-6)
