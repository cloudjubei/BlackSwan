"""Regression test for the Agent57 intrinsic-reward sampling fix. CustomAgent57ReplayBuffer must record
the buffer positions it samples (last_sampled_indices) so Agent57.compute_intrinsic_rewards can look up the
per-sample bonus — SB3's ReplayBufferSamples carries no `indices` (the original AttributeError)."""
import numpy as np
from gymnasium import spaces

from src.model.custom.agent57.agent57 import CustomAgent57ReplayBuffer


def _buf(obs_dim=4, size=10):
    return CustomAgent57ReplayBuffer(
        buffer_size=size,
        observation_space=spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32),
        action_space=spaces.Discrete(3),
        device="cpu",
    )


def _add(buf, n, obs_dim=4):
    for i in range(n):
        obs = np.full((1, obs_dim), float(i), dtype=np.float32)
        buf.add(obs, obs + 0.5, np.array([[i % 3]]), np.array([1.0]), np.array([False]), [{}])


def test_sample_records_the_sampled_positions():
    buf = _buf()
    _add(buf, 8)
    buf.last_sampled_indices = None
    buf.sample(4)
    assert buf.last_sampled_indices is not None
    assert len(buf.last_sampled_indices) == 4
    assert all(0 <= i < buf.pos for i in buf.last_sampled_indices)
    # the recorded positions index intrinsic_rewards exactly as compute_intrinsic_rewards does
    assert buf.intrinsic_rewards[buf.last_sampled_indices].shape == (4,)


def test_add_writes_a_positive_intrinsic_reward_per_position():
    buf = _buf()
    _add(buf, 5)
    # add() stores a per-position intrinsic bonus (count + prediction-error + novelty, all >= 0 with a
    # strictly-positive count term) at pos-1 each call.
    assert np.all(buf.intrinsic_rewards[:5] > 0)
