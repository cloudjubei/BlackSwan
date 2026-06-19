"""Direct tests for the dip-PREDICTION env's classification bookkeeping + reward scoring.

DipPredictEnv is a 2-action (0=nothing, 1=dip) classification-style env. Its reward and the
recall/precision/negative-recall/accuracy/streak bookkeeping are pure functions of:
  - the data provider's get_signal_buy_profitable(step) vs the provider's buyreward_maxwait, and
  - the agent's action (0/1).
A step is a real "dip" (buy opportunity) when profitable_steps <= buyreward_maxwait.

The heavy __init__ (which calls reset() and touches the data provider's real values) is bypassed
with Cls.__new__; we set only the attributes each method-under-test reads and fake the provider with
a tiny stub, mirroring test_base_crypto_env.py / test_trade_all_crypto_env.py.
"""

import types

import numpy as np
import pytest

from src.environment.dip_predict_env import DipPredictEnv

# Distinct values so each reward branch is identifiable by its return value alone.
_MULTIPLIERS = {
    "combo_buy": 5.0,        # correct dip (predicted dip, was a dip)
    "combo_wrongaction": -1.0,  # missed dip (predicted nothing, was a dip)
    "combo_sell": -2.0,      # false alarm (predicted dip, was NOT a dip)
    "combo_noaction": 0.3,   # correct reject (predicted nothing, was NOT a dip)
}


class FakeProvider:
    """Minimal provider: per-step profitable-wait values + a maxwait threshold and lookback."""

    def __init__(self, profitable_by_step, *, maxwait=3, lookback=1, values=None, prices=None):
        # profitable_by_step[step] = "steps until profitable"; <= maxwait means the step is a dip.
        self._profitable = list(profitable_by_step)
        self.buyreward_maxwait = maxwait
        self._lookback = lookback
        # get_values returns either a flat 1-D array (lookback==1) or a 2-D grid (lookback>1).
        self._values = values
        self._prices = prices

    def get_timesteps(self):
        return len(self._profitable)

    def get_lookback_window(self):
        return self._lookback

    def get_price(self, step):
        if self._prices is not None:
            return self._prices[step]
        return 100.0

    def get_signal_buy_profitable(self, step):
        return self._profitable[step]

    def get_values(self, step):
        if self._values is not None:
            return self._values[step]
        return np.zeros(2, dtype=np.float32)


def _env(provider, multipliers=None):
    """Construct a DipPredictEnv without running the heavy __init__, then prime reset() state."""
    e = DipPredictEnv.__new__(DipPredictEnv)
    e.data_provider = provider
    e.reward_multipliers = dict(multipliers if multipliers is not None else _MULTIPLIERS)
    # reset() initialises every counter/history list the methods below read.
    e.reset()
    return e


# --- _calculate_reward: the four classification branches ---


def test_reward_correct_dip_predicted():
    # profitable wait 2 <= maxwait 3 -> a dip; action 1 (predict dip) -> correct.
    e = _env(FakeProvider([2], maxwait=3))
    e.current_step = 0
    r = e._calculate_reward(1)
    assert r == _MULTIPLIERS["combo_buy"]
    assert e.correct_dips == 1
    assert e.total_dips_seen == 1
    assert e.incorrect_dips == 0
    assert e.total_nondips_seen == 0


def test_reward_missed_dip():
    # a dip (wait 1 <= 3) but action 0 (predicted nothing) -> wrong action, no correct credit.
    e = _env(FakeProvider([1], maxwait=3))
    e.current_step = 0
    r = e._calculate_reward(0)
    assert r == _MULTIPLIERS["combo_wrongaction"]
    assert e.total_dips_seen == 1
    assert e.correct_dips == 0


def test_reward_false_alarm_predicted_dip_on_nondip():
    # NOT a dip (wait 5 > 3) but action 1 -> false positive: counts incorrect_dips, returns combo_sell.
    e = _env(FakeProvider([5], maxwait=3))
    e.current_step = 0
    r = e._calculate_reward(1)
    assert r == _MULTIPLIERS["combo_sell"]
    assert e.total_nondips_seen == 1
    assert e.incorrect_dips == 1
    assert e.total_dips_seen == 0


def test_reward_correct_reject_nondip():
    # NOT a dip, action 0 -> correct reject: combo_noaction, no incorrect credit.
    e = _env(FakeProvider([9], maxwait=3))
    e.current_step = 0
    r = e._calculate_reward(0)
    assert r == _MULTIPLIERS["combo_noaction"]
    assert e.total_nondips_seen == 1
    assert e.incorrect_dips == 0


def test_reward_dip_boundary_equal_to_maxwait_is_a_dip():
    # profitable_steps == buyreward_maxwait is INCLUSIVE (<=), so it must count as a dip.
    e = _env(FakeProvider([3], maxwait=3))
    e.current_step = 0
    r = e._calculate_reward(1)
    assert r == _MULTIPLIERS["combo_buy"]
    assert e.total_dips_seen == 1


def test_reward_just_over_maxwait_is_a_nondip():
    e = _env(FakeProvider([4], maxwait=3))
    e.current_step = 0
    r = e._calculate_reward(0)
    assert r == _MULTIPLIERS["combo_noaction"]
    assert e.total_nondips_seen == 1


# --- recall / precision / negative-recall / accuracy ---


def test_metrics_default_to_one_when_no_samples_seen():
    # Fresh env (nothing seen): all rate-style metrics default to 1 (vacuously perfect).
    e = _env(FakeProvider([1], maxwait=3))
    assert e._get_recall() == 1
    assert e._get_precision() == 1
    assert e._get_negative_recall() == 1
    assert e._get_accuracy() == 1


def test_recall_is_correct_over_total_dips():
    e = _env(FakeProvider([1], maxwait=3))
    e.correct_dips = 3
    e.total_dips_seen = 4
    assert e._get_recall() == 3 / 4


def test_precision_is_correct_over_all_guessed_dips():
    e = _env(FakeProvider([1], maxwait=3))
    e.correct_dips = 3
    e.incorrect_dips = 1
    assert e._get_precision() == 3 / 4


def test_negative_recall_penalizes_false_alarms():
    # 2 false alarms out of 10 non-dips -> 1 - 0.2 = 0.8.
    e = _env(FakeProvider([1], maxwait=3))
    e.incorrect_dips = 2
    e.total_nondips_seen = 10
    assert e._get_negative_recall() == pytest.approx(0.8)


def test_accuracy_is_mean_of_recall_and_negative_recall():
    e = _env(FakeProvider([1], maxwait=3))
    e.correct_dips = 1
    e.total_dips_seen = 2          # recall 0.5
    e.incorrect_dips = 1
    e.total_nondips_seen = 4       # neg-recall 0.75
    assert e._get_accuracy() == pytest.approx((0.5 + 0.75) / 2)


# --- _update_streaks: it both records the metric snapshots and runs the streak machine ---


def test_update_streaks_appends_a_metric_snapshot_each_call():
    e = _env(FakeProvider([1], maxwait=3))
    e._update_streaks(action=0, reward=0.0, done=False)
    assert len(e.recalls) == 1
    assert len(e.precisions) == 1
    assert len(e.negative_recalls) == 1
    assert len(e.accuracies) == 1


def test_streak_increments_on_rewarded_dip_prediction():
    e = _env(FakeProvider([1], maxwait=3))
    e._update_streaks(action=1, reward=5.0, done=False)
    e._update_streaks(action=1, reward=5.0, done=False)
    assert e.current_streak == 2
    # Not done and still on a winning run -> nothing finalised into streaks yet.
    assert e.streaks == []


def test_streak_breaks_and_records_on_unrewarded_dip_prediction():
    e = _env(FakeProvider([1], maxwait=3))
    e.current_streak = 4
    e._update_streaks(action=1, reward=-2.0, done=False)  # a false alarm breaks the run
    assert e.streaks == [4]
    assert e.current_streak == 0


def test_streak_break_from_zero_records_no_spurious_zero():
    # Breaking (unrewarded dip prediction) while current_streak is already 0 must NOT append a 0.
    # A back-to-back / first-step false alarm should leave streaks empty, otherwise the avg-streak
    # metric in get_run_state is biased downward by phantom zeros.
    e = _env(FakeProvider([1], maxwait=3))
    assert e.current_streak == 0
    e._update_streaks(action=1, reward=-2.0, done=False)  # break from 0
    e._update_streaks(action=1, reward=-2.0, done=False)  # break from 0 again
    assert e.streaks == []
    assert e.current_streak == 0
    # A genuine run still records when it breaks.
    e.current_streak = 3
    e._update_streaks(action=1, reward=-2.0, done=False)
    assert e.streaks == [3]


def test_streak_done_with_zero_streak_records_no_spurious_zero():
    # Reaching done with no active streak (hold path, or winning path that never ran) records nothing.
    e_hold = _env(FakeProvider([1], maxwait=3))
    e_hold.current_streak = 0
    e_hold._update_streaks(action=0, reward=0.3, done=True)  # elif-done with zero streak
    assert e_hold.streaks == []


def test_avg_streak_not_polluted_by_break_from_zero():
    # Two spurious break-from-zero events + one real 4-run: avg-streak should be 4.0, not 4/3.
    e = _env(FakeProvider([1], maxwait=3))
    e._update_streaks(action=1, reward=-2.0, done=False)  # break from 0 -> no record
    e._update_streaks(action=1, reward=-2.0, done=False)  # break from 0 -> no record
    e.current_streak = 4
    e._update_streaks(action=1, reward=-2.0, done=False)  # real run breaks -> records 4
    assert e.streaks == [4]
    state = e.get_run_state()
    assert state[7] == pytest.approx(4.0)  # avg_streak = 4/1, not diluted by zeros
    assert state[8] == 4                   # max_streak


def test_streak_finalised_on_done_while_still_winning():
    e = _env(FakeProvider([1], maxwait=3))
    e._update_streaks(action=1, reward=5.0, done=False)  # streak -> 1
    e._update_streaks(action=1, reward=5.0, done=True)   # streak -> 2 then recorded
    assert e.current_streak == 2
    assert e.streaks == [2]


def test_streak_recorded_on_done_with_hold_action():
    # action != 1 on the final step: the elif-done branch records the current streak as-is.
    e = _env(FakeProvider([1], maxwait=3))
    e.current_streak = 7
    e._update_streaks(action=0, reward=0.3, done=True)
    assert e.streaks == [7]
    assert e.current_streak == 7  # hold path does not reset


def test_streak_hold_midrun_does_not_touch_streak():
    e = _env(FakeProvider([1], maxwait=3))
    e.current_streak = 2
    e._update_streaks(action=0, reward=0.3, done=False)
    assert e.current_streak == 2
    assert e.streaks == []


# --- update_reward: accumulation + history ---


def test_update_reward_accumulates_total_and_history():
    e = _env(FakeProvider([2, 5], maxwait=3))
    e.current_step = 0
    r1 = e.update_reward(1)  # correct dip -> +5
    e.current_step = 1
    r2 = e.update_reward(1)  # false alarm -> -2
    assert r1 == 5.0 and r2 == -2.0
    assert e.total_reward == pytest.approx(3.0)
    assert e.rewards_history == [5.0, -2.0]
    assert e.rewards_totals == [5.0, 3.0]  # running totals


# --- resolve_action ---


def test_resolve_action_records_the_raw_action():
    e = _env(FakeProvider([1], maxwait=3))
    e.resolve_action(1)
    e.resolve_action(0)
    assert e.actions == [1, 0]


# --- step: end-to-end single transition + done flag ---


def test_step_advances_and_reports_reward_and_not_done():
    # Two timesteps -> after one step current_step==1 < 2 timesteps so not done.
    e = _env(FakeProvider([2, 5], maxwait=3))
    obs, reward, done, finished_early, info = e.step(1)  # step 0 is a dip, predicted -> +5
    assert reward == 5.0
    assert e.current_step == 1
    assert done is False
    assert finished_early is False
    assert info == {}
    assert e.actions == [1]
    assert e.correct_dips == 1


def test_step_sets_done_on_last_timestep():
    e = _env(FakeProvider([2], maxwait=3))  # a single timestep
    _, _, done, _, _ = e.step(1)
    assert e.current_step == 1
    assert done is True
    # done + winning dip prediction -> streak recorded.
    assert e.streaks == [1]


def test_step_unwraps_ndarray_action():
    # RL libs can hand an array action; step must .item() it before comparing to 1.
    e = _env(FakeProvider([2, 2], maxwait=3))
    _, reward, _, _, _ = e.step(np.array([1]))
    assert reward == _MULTIPLIERS["combo_buy"]
    assert e.actions == [1]  # stored as the unwrapped scalar


def test_step_records_current_price_from_provider():
    e = _env(FakeProvider([2, 2], maxwait=3, prices=[111.0, 222.0]))
    e.step(0)
    assert e.current_price == 111.0


# --- get_next_observation: flatten only when lookback>1 ---


def test_get_next_observation_flat_when_lookback_one():
    vals = [np.zeros(3, dtype=np.float32)]
    e = _env(FakeProvider([1], maxwait=3, lookback=1, values=vals))
    e.current_step = 0
    out = e.get_next_observation()
    assert out.shape == (3,)


def test_get_next_observation_flattens_grid_when_lookback_gt_one():
    grid = np.arange(6, dtype=np.float32).reshape(2, 3)  # [lookback=2, features=3]
    e = _env(FakeProvider([1], maxwait=3, lookback=2, values=[grid]))
    e.current_step = 0
    out = e.get_next_observation()
    assert out.shape == (6,)  # flattened
    assert np.array_equal(out, grid.flatten())


# --- get_run_state: derived f1 / ratio / streak summary ---


def test_get_run_state_zeros_when_nothing_happened():
    e = _env(FakeProvider([1], maxwait=3))
    state = e.get_run_state()
    # total_reward, f1, dips_ratio, accuracy, precision, recall, neg_recall, avg_streak, max_streak
    assert state[0] == 0          # total_reward
    assert state[1] == 0          # f1 (precision+recall==0 guard)
    assert state[2] == 0          # dips_ratio == correct_dips (0) when no incorrect
    assert state[7] == 0          # avg_streak (no streaks)
    assert state[8] == 0          # max_streak (no streaks)
    assert isinstance(state[-1], str) and state[-1] != ""  # reward multipliers string


def test_get_run_state_computes_f1_and_dips_ratio_and_streaks():
    e = _env(FakeProvider([1], maxwait=3))
    # Drive precision/recall: 3 correct dips of 4 dips, 1 false alarm of 8 non-dips.
    e.correct_dips = 3
    e.total_dips_seen = 4
    e.incorrect_dips = 1
    e.total_nondips_seen = 8
    e.precisions = [e._get_precision()]
    e.recalls = [e._get_recall()]
    e.negative_recalls = [e._get_negative_recall()]
    e.accuracies = [e._get_accuracy()]
    e.streaks = [2, 4]
    e.total_reward = 12.5
    state = e.get_run_state()
    precision = 3 / 4
    recall = 3 / 4
    expected_f1 = 2 * precision * recall / (precision + recall)
    assert state[0] == 12.5
    assert state[1] == pytest.approx(expected_f1)
    assert state[2] == pytest.approx(3 / 1)  # correct/incorrect
    assert state[7] == pytest.approx(3.0)    # avg streak (2+4)/2
    assert state[8] == 4                     # max streak
    assert state[9] == "[3/4]-[1/8]"


def test_get_run_state_dips_ratio_falls_back_to_correct_when_no_incorrect():
    e = _env(FakeProvider([1], maxwait=3))
    e.correct_dips = 6
    e.incorrect_dips = 0
    state = e.get_run_state()
    assert state[2] == 6  # divides-by-zero guard: returns correct_dips itself
