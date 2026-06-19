"""Direct unit tests for TrendPredictEnv reward/scoring + step bookkeeping.

The heavy __init__ (which calls reset() and builds gym spaces off a real data provider) is
bypassed via ``__new__``; only the attributes each method-under-test reads are set, with a tiny
fake data provider. No GPU / training loop / data files involved.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.environment.trend_predict_env import TrendPredictEnv


class _FakeProvider:
    """Minimal data provider: signals/prices/values are looked up per step from dicts/lists."""

    def __init__(self, signals=None, prices=None, values=None, lookback=0, timesteps=None):
        self._signals = signals or {}
        self._prices = prices or {}
        self._values = values or {}
        self._lookback = lookback
        self._timesteps = timesteps if timesteps is not None else 10

    def get_signal_buy_sell(self, step):
        return self._signals.get(step, 0)

    def get_price(self, step):
        return self._prices.get(step, 0.0)

    def get_values(self, step):
        return self._values.get(step, np.zeros(3, dtype=np.float32))

    def get_lookback_window(self):
        return self._lookback

    def get_timesteps(self):
        return self._timesteps


def _env(provider=None):
    """Construct a TrendPredictEnv without running __init__, wired to a fake provider."""
    e = TrendPredictEnv.__new__(TrendPredictEnv)
    e.data_provider = provider if provider is not None else _FakeProvider()
    e.current_step = 0
    e.current_price = 0
    e.current_streak = 0
    e.total_reward = 0
    e.rewards_history = []
    e.actions = []
    e.rewards_totals = []
    e.accuracies = []
    e.streaks = []
    return e


# ---------------------------------------------------------------------------
# _calculate_reward: the core trend-classification reward.
# action == 1 means "predict Negative"; anything else means "predict Positive".
# ---------------------------------------------------------------------------

def test_calculate_reward_negative_prediction_correct():
    # action 1 (predict Negative) rewarded when the signal is negative (a sell/down trend).
    e = _env(_FakeProvider(signals={0: -3}))
    assert e._calculate_reward(1) == 1


def test_calculate_reward_negative_prediction_wrong_when_signal_positive():
    e = _env(_FakeProvider(signals={0: 2}))
    assert e._calculate_reward(1) == 0


def test_calculate_reward_positive_prediction_correct():
    # action 0 (predict Positive) rewarded when the signal is positive (a buy/up trend).
    e = _env(_FakeProvider(signals={0: 5}))
    assert e._calculate_reward(0) == 1


def test_calculate_reward_positive_prediction_wrong_when_signal_negative():
    e = _env(_FakeProvider(signals={0: -1}))
    assert e._calculate_reward(0) == 0


def test_calculate_reward_zero_signal_is_never_rewarded():
    # signal == 0 is neither >0 nor <0 -> both predictions score 0.
    e = _env(_FakeProvider(signals={0: 0}))
    assert e._calculate_reward(0) == 0
    assert e._calculate_reward(1) == 0


def test_calculate_reward_reads_signal_at_current_step():
    e = _env(_FakeProvider(signals={0: 2, 3: -2}))
    e.current_step = 3
    assert e._calculate_reward(1) == 1  # step-3 signal is negative
    assert e._calculate_reward(0) == 0


# ---------------------------------------------------------------------------
# update_reward: accumulation bookkeeping.
# ---------------------------------------------------------------------------

def test_update_reward_accumulates_total_and_histories():
    e = _env(_FakeProvider(signals={0: 5}))
    r = e.update_reward(0)  # correct positive -> reward 1
    assert r == 1
    assert e.total_reward == 1
    assert e.rewards_history == [1]
    assert e.rewards_totals == [1]


def test_update_reward_running_total_across_steps():
    e = _env(_FakeProvider(signals={0: 5}))
    e.update_reward(0)  # +1
    e.current_step = 0  # still reads signal 5 -> another +1
    e.update_reward(0)
    assert e.total_reward == 2
    assert e.rewards_history == [1, 1]
    assert e.rewards_totals == [1, 2]  # cumulative snapshots


# ---------------------------------------------------------------------------
# resolve_action: records the action.
# ---------------------------------------------------------------------------

def test_resolve_action_appends_action():
    e = _env()
    e.resolve_action(1)
    e.resolve_action(0)
    assert e.actions == [1, 0]


# ---------------------------------------------------------------------------
# _get_accuracy: total_reward / number of rewards seen.
# ---------------------------------------------------------------------------

def test_get_accuracy_is_total_reward_over_count():
    e = _env()
    e.total_reward = 3
    e.rewards_history = [1, 0, 1, 1]
    assert e._get_accuracy() == 3 / 4


# ---------------------------------------------------------------------------
# _update_streaks: streak tracking + accuracy snapshotting.
# ---------------------------------------------------------------------------

def test_update_streaks_positive_reward_increments_streak_no_append_until_done():
    e = _env()
    # one prior reward recorded so _get_accuracy has a denominator.
    e.total_reward = 1
    e.rewards_history = [1]
    e._update_streaks(action=0, reward=1, done=False)
    assert e.current_streak == 1
    assert e.streaks == []  # streak only flushed on done or on a miss
    assert e.accuracies == [1.0]


def test_update_streaks_positive_reward_on_done_appends_streak():
    e = _env()
    e.total_reward = 2
    e.rewards_history = [1, 1]
    e.current_streak = 1
    e._update_streaks(action=0, reward=1, done=True)
    assert e.current_streak == 2
    assert e.streaks == [2]


def test_update_streaks_miss_flushes_and_resets():
    e = _env()
    e.total_reward = 2
    e.rewards_history = [1, 1, 0]
    e.current_streak = 2
    e._update_streaks(action=0, reward=0, done=False)
    assert e.streaks == [2]
    assert e.current_streak == 0


def test_update_streaks_appends_accuracy_each_call():
    e = _env()
    e.total_reward = 1
    e.rewards_history = [1, 1]
    e._update_streaks(action=0, reward=1, done=False)
    assert e.accuracies == [0.5]


# ---------------------------------------------------------------------------
# step: full bookkeeping orchestration.
# ---------------------------------------------------------------------------

def test_step_advances_and_returns_tuple():
    prov = _FakeProvider(signals={0: 5}, prices={0: 123.0}, timesteps=3,
                         values={1: np.array([1.0, 2.0, 3.0], dtype=np.float32)})
    e = _env(prov)
    obs, reward, done, finished_early, info = e.step(0)
    assert reward == 1  # correct positive prediction at step 0
    assert e.current_step == 1
    assert done is False  # 1 < 3
    assert finished_early is False
    assert info == {}
    assert e.current_price == 123.0
    assert e.actions == [0]
    np.testing.assert_array_equal(obs, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_step_marks_done_at_final_timestep():
    prov = _FakeProvider(signals={0: 5}, prices={0: 1.0}, timesteps=1)
    e = _env(prov)
    _, _, done, _, _ = e.step(0)
    assert e.current_step == 1
    assert done is True  # 1 >= 1


def test_step_unwraps_numpy_array_action():
    # RL models hand back actions as ndarrays; step() must unwrap to a scalar before comparison.
    prov = _FakeProvider(signals={0: -3}, prices={0: 1.0}, timesteps=5)
    e = _env(prov)
    _, reward, _, _, _ = e.step(np.array([1]))
    assert reward == 1  # action 1 vs negative signal -> correct
    assert e.actions == [1]  # stored as the unwrapped scalar


def test_step_done_triggers_streak_flush():
    # On the terminal step a positive reward must push the streak onto streaks via _update_streaks.
    prov = _FakeProvider(signals={0: 5}, prices={0: 1.0}, timesteps=1)
    e = _env(prov)
    e.step(0)
    assert e.streaks == [1]


# ---------------------------------------------------------------------------
# get_next_observation: flatten only when lookback > 1.
# ---------------------------------------------------------------------------

def test_get_next_observation_no_flatten_when_lookback_le_1():
    grid = np.arange(6, dtype=np.float32).reshape(2, 3)
    prov = _FakeProvider(values={0: grid}, lookback=1)
    e = _env(prov)
    out = e.get_next_observation()
    assert out.shape == (2, 3)


def test_get_next_observation_flattens_when_lookback_gt_1():
    grid = np.arange(6, dtype=np.float32).reshape(2, 3)
    prov = _FakeProvider(values={0: grid}, lookback=2)
    e = _env(prov)
    out = e.get_next_observation()
    assert out.shape == (6,)


# ---------------------------------------------------------------------------
# get_run_state: summary metrics + multiplier formatting.
# ---------------------------------------------------------------------------

def test_get_run_state_empty_lists_give_zeroes():
    e = _env()
    e.total_reward = 0
    e.reward_multipliers = {}
    state = e.get_run_state()
    assert state[0] == 0  # total_reward
    assert state[1] == 0  # accuracy (empty)
    assert state[2] == 0  # avg_streak (empty)
    assert state[3] == 0  # max_streak (empty)
    assert state[4] == ""  # no multipliers


def test_get_run_state_computes_streak_and_accuracy_stats():
    e = _env()
    e.total_reward = 4
    e.accuracies = [0.4, 0.5, 0.6]
    e.streaks = [1, 3, 2]
    e.reward_multipliers = {"a": 1.5, "b": 2.0}
    state = e.get_run_state()
    assert state[0] == 4
    assert state[1] == 0.6  # last accuracy
    assert state[2] == pytest.approx(2.0)  # mean of [1,3,2]
    assert state[3] == 3  # max
    assert state[4] == "1.500;2.000;"  # formatted multipliers


# ---------------------------------------------------------------------------
# create_action_space / create_observation_space: space shapes.
# ---------------------------------------------------------------------------

def test_create_action_space_is_binary():
    e = _env()
    space = e.create_action_space()
    assert space.n == 2


def test_create_observation_space_matches_obs_shape():
    e = _env()
    obs = np.zeros((7,), dtype=np.float32)
    space = e.create_observation_space(obs)
    assert space.shape == (7,)
    assert space.low.min() == -1 and space.high.max() == 1


# ---------------------------------------------------------------------------
# reset: list (re)initialisation + actions_epochs rollover.
# ---------------------------------------------------------------------------

def test_reset_initialises_state_and_returns_obs():
    prov = _FakeProvider(values={0: np.array([9.0, 8.0], dtype=np.float32)}, lookback=0)
    e = TrendPredictEnv.__new__(TrendPredictEnv)
    e.data_provider = prov
    obs, info = e.reset()
    assert info == {}
    assert e.current_step == 0
    assert e.current_price == 0
    assert e.current_streak == 0
    assert e.total_reward == 0
    assert e.rewards_history == [] and e.actions == [] and e.rewards_totals == []
    assert e.accuracies == [] and e.streaks == []
    assert e.actions_epochs == []  # first reset seeds the epochs list
    np.testing.assert_array_equal(obs, np.array([9.0, 8.0], dtype=np.float32))


def test_reset_second_time_rolls_prior_actions_into_epochs():
    prov = _FakeProvider(values={0: np.zeros(2, dtype=np.float32)})
    e = TrendPredictEnv.__new__(TrendPredictEnv)
    e.data_provider = prov
    e.reset()
    e.actions = [0, 1, 1]  # simulate a completed episode
    e.reset()
    assert e.actions_epochs == [[0, 1, 1]]  # prior actions archived
    assert e.actions == []  # fresh episode buffer
