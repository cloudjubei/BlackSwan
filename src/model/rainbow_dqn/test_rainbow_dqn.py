"""Direct regression tests for RainbowDQN.train's bookkeeping (logging-only fixes).

RainbowDQN subclasses SB3's DQN and reimplements train() to use the prioritized replay
buffer. Two logging-only bugs were fixed: (1) self._n_updates was incremented twice per
call (once per gradient step inside the loop AND once by gradient_steps after it), and
(2) the per-step loss was never collected, so train/loss logged np.mean([]) -> nan. We
build a tiny RainbowDQN on CartPole (CPU, no training loop), seed its buffer, and call
train() directly — no GPU, no market data, no learn() loop.
"""

import os
import sys

import gymnasium as gym
import numpy as np
from stable_baselines3.common.logger import configure

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.model.rainbow_dqn.rainbow_dqn import RainbowDQN


def _seeded_model():
    env = gym.make("CartPole-v1")  # Box(4) obs + Discrete(2) action — a minimal DQN-compatible env
    model = RainbowDQN(env=env, buffer_size=256, learning_starts=0, batch_size=8, device="cpu", seed=0)
    model.set_logger(configure(folder=None, format_strings=[]))  # records to name_to_value, no output
    obs, _ = env.reset(seed=0)
    for _ in range(32):  # seed the replay buffer so sample() has data
        action = int(env.action_space.sample())
        next_obs, reward, term, trunc, info = env.step(action)
        model.replay_buffer.add(
            np.array([obs], dtype=np.float32),
            np.array([next_obs], dtype=np.float32),
            np.array([[action]]),
            np.array([reward], dtype=np.float32),
            np.array([term], dtype=np.float32),
            [info],
        )
        obs = env.reset(seed=0)[0] if (term or trunc) else next_obs
    return model


def test_train_increments_n_updates_by_gradient_steps_only():
    # The duplicate per-iteration `self._n_updates += 1` was removed; only the single
    # `+= gradient_steps` after the loop remains (the counter was advancing 2x gradient_steps).
    model = _seeded_model()
    before = model._n_updates
    model.train(gradient_steps=3, batch_size=8)
    assert model._n_updates - before == 3


def test_train_logs_finite_loss():
    # Per-step losses are now collected, so train/loss is a real finite mean (was np.mean([]) -> nan).
    model = _seeded_model()
    model.train(gradient_steps=3, batch_size=8)
    logged = model.logger.name_to_value.get("train/loss")
    assert logged is not None
    assert np.isfinite(logged)
