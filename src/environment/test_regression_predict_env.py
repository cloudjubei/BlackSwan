"""Direct unit tests for RegressionPredictEnv result-tracking + metric logic.

The heavy __init__ (which iterates the data provider, builds a torch DataLoader and calls reset())
is bypassed via ``__new__``; only the attributes each method-under-test reads are set. Predictions
and actuals are fed as small numpy arrays (they only need ``==``/``&``/``.sum().item()``/``len()``).
No GPU / training loop / data files involved.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.environment.regression_predict_env import RegressionPredictEnv


class _FakeProvider:
    def __init__(self, values=None, lookback=0, timesteps=10):
        self._values = values or {}
        self._lookback = lookback
        self._timesteps = timesteps

    def get_values(self, step):
        return self._values.get(step, np.zeros(3, dtype=np.float32))

    def get_lookback_window(self):
        return self._lookback

    def get_timesteps(self):
        return self._timesteps


def _env(provider=None):
    """Construct a RegressionPredictEnv without __init__, with counters zeroed (mirrors reset())."""
    e = RegressionPredictEnv.__new__(RegressionPredictEnv)
    e.data_provider = provider if provider is not None else _FakeProvider()
    e.current_step = 0
    e.total_seen = 0
    e.total_correct = 0
    e.total_incorrect = 0
    e.correct_pos = 0
    e.incorrect_pos = 0
    e.correct_neg = 0
    e.incorrect_neg = 0
    e.predictions = []
    e.actuals = []
    return e


# ---------------------------------------------------------------------------
# store_result: confusion-matrix accumulation from prediction/actual tensors.
# The real caller (regression_model.test) hands in torch tensors, so that is the contract:
# they must support ==, &, .sum().item() and be iterable for the `predictions += prediction`
# list extension. Class 1 = positive; "total_correct" counts positive actuals.
# ---------------------------------------------------------------------------

def test_store_result_records_predictions_and_actuals():
    e = _env()
    actual = torch.tensor([1, 0, 1, 0])
    pred = torch.tensor([1, 0, 1, 0])
    e.store_result(pred, actual)
    # `predictions += prediction` extends the list with per-element scalar tensors.
    assert [int(p) for p in e.predictions] == [1, 0, 1, 0]
    assert [int(a) for a in e.actuals] == [1, 0, 1, 0]


def test_store_result_counts_confusion_matrix():
    e = _env()
    actual = torch.tensor([1, 0, 1, 0, 1])  # 3 positives, 2 negatives
    pred = torch.tensor([1, 1, 0, 0, 1])    # predicts pos for idx 0,1,4
    e.store_result(pred, actual)
    assert e.total_seen == 5
    assert e.total_correct == 3   # positive actuals
    assert e.total_incorrect == 2  # negative actuals
    # prediction_correct (pred==1) & actual_correct (actual==1): idx 0 and 4 -> 2 true positives
    assert e.correct_pos == 2
    # pred==1 & actual==0: idx 1 -> 1 false positive
    assert e.incorrect_pos == 1
    # pred==0 & actual==0: idx 3 -> 1 true negative
    assert e.correct_neg == 1
    # pred==0 & actual==1: idx 2 -> 1 false negative
    assert e.incorrect_neg == 1


def test_store_result_accumulates_across_calls():
    e = _env()
    e.store_result(torch.tensor([1, 0]), torch.tensor([1, 0]))
    e.store_result(torch.tensor([1, 1]), torch.tensor([0, 1]))
    assert e.total_seen == 4
    assert e.correct_pos == 2  # idx0 first batch + idx1 second batch
    assert e.incorrect_pos == 1  # idx0 second batch
    assert e.correct_neg == 1  # idx1 first batch


# ---------------------------------------------------------------------------
# _get_recall = TP / positive-actuals, with empty-guard returning 1.
# ---------------------------------------------------------------------------

def test_recall_normal():
    e = _env()
    e.correct_pos = 3
    e.total_correct = 4
    assert e._get_recall() == 3 / 4


def test_recall_guard_when_no_positives():
    e = _env()
    e.correct_pos = 0
    e.total_correct = 0
    assert e._get_recall() == 1


# ---------------------------------------------------------------------------
# _get_precision = TP / (TP + FP), guard returns 1.
# ---------------------------------------------------------------------------

def test_precision_normal():
    e = _env()
    e.correct_pos = 3
    e.incorrect_pos = 1
    assert e._get_precision() == 3 / 4


def test_precision_guard_when_nothing_guessed_positive():
    e = _env()
    e.correct_pos = 0
    e.incorrect_pos = 0
    assert e._get_precision() == 1


# ---------------------------------------------------------------------------
# _get_negative_recall = 1 - FP/negative-actuals, guard returns 1.
# ---------------------------------------------------------------------------

def test_negative_recall_normal():
    e = _env()
    e.incorrect_pos = 1
    e.total_incorrect = 4
    assert e._get_negative_recall() == 1 - 1 / 4


def test_negative_recall_guard_when_no_negatives():
    e = _env()
    e.incorrect_pos = 0
    e.total_incorrect = 0
    assert e._get_negative_recall() == 1


def test_negative_recall_zero_when_all_negatives_misclassified():
    e = _env()
    e.incorrect_pos = 4
    e.total_incorrect = 4
    assert e._get_negative_recall() == 0


# ---------------------------------------------------------------------------
# _get_accuracy = mean(recall, negative_recall) (balanced accuracy).
# ---------------------------------------------------------------------------

def test_accuracy_is_balanced_mean():
    e = _env()
    e.correct_pos = 3
    e.total_correct = 4   # recall 0.75
    e.incorrect_pos = 1
    e.total_incorrect = 4  # neg recall 0.75
    assert e._get_accuracy() == pytest.approx(0.75)


def test_accuracy_perfect_classifier_is_one():
    e = _env()
    e.correct_pos = 5
    e.total_correct = 5
    e.incorrect_pos = 0
    e.total_incorrect = 5
    assert e._get_accuracy() == 1.0


# ---------------------------------------------------------------------------
# get_run_state: f1, simple_ratio, and the formatted confusion string.
# ---------------------------------------------------------------------------

def test_get_run_state_full_metrics():
    e = _env()
    e.correct_pos = 3
    e.incorrect_pos = 1
    e.total_correct = 4
    e.total_incorrect = 4
    state = e.get_run_state()
    precision = 3 / 4
    recall = 3 / 4
    neg_recall = 1 - 1 / 4
    f1 = 2 * precision * recall / (precision + recall)
    simple_ratio = 3 / 1
    assert state[0] == pytest.approx(f1)
    assert state[1] == pytest.approx(simple_ratio)
    assert state[2] == pytest.approx((recall + neg_recall) / 2)
    assert state[3] == pytest.approx(precision)
    assert state[4] == pytest.approx(recall)
    assert state[5] == pytest.approx(neg_recall)
    assert state[6] == "[3/1]-[4/4]"


def test_get_run_state_simple_ratio_falls_back_to_correct_pos_without_false_positives():
    # incorrect_pos == 0 -> avoid div-by-zero by returning correct_pos directly.
    e = _env()
    e.correct_pos = 5
    e.incorrect_pos = 0
    e.total_correct = 5
    e.total_incorrect = 5
    state = e.get_run_state()
    assert state[1] == 5


def test_get_run_state_f1_zero_when_precision_and_recall_zero():
    # No true positives at all and no guesses -> precision/recall guards both return 1, so
    # f1 is well-defined; to force the f1==0 branch we need precision+recall == 0, which only
    # happens when correct_pos==0 AND there ARE positives/false-positives.
    e = _env()
    e.correct_pos = 0
    e.incorrect_pos = 2  # precision = 0/2 = 0
    e.total_correct = 3  # recall = 0/3 = 0
    e.total_incorrect = 5
    state = e.get_run_state()
    assert state[0] == 0  # f1 guarded to 0 when precision+recall == 0


# ---------------------------------------------------------------------------
# step: pure step counter + done flag (this env's step does no trading work).
# ---------------------------------------------------------------------------

def test_step_advances_and_not_done():
    prov = _FakeProvider(timesteps=3)
    e = _env(prov)
    obs, reward, done, finished_early, info = e.step(0)
    assert obs == [] and reward == 0
    assert e.current_step == 1
    assert done is False
    assert finished_early is False
    assert info == {}


def test_step_done_at_final_timestep():
    prov = _FakeProvider(timesteps=1)
    e = _env(prov)
    _, _, done, _, _ = e.step(0)
    assert e.current_step == 1
    assert done is True


# ---------------------------------------------------------------------------
# get_next_observation: flatten only when lookback > 1.
# ---------------------------------------------------------------------------

def test_get_next_observation_no_flatten_when_lookback_le_1():
    grid = np.arange(6, dtype=np.float32).reshape(2, 3)
    e = _env(_FakeProvider(values={0: grid}, lookback=1))
    assert e.get_next_observation().shape == (2, 3)


def test_get_next_observation_flattens_when_lookback_gt_1():
    grid = np.arange(6, dtype=np.float32).reshape(2, 3)
    e = _env(_FakeProvider(values={0: grid}, lookback=2))
    assert e.get_next_observation().shape == (6,)


# ---------------------------------------------------------------------------
# create_action_space.
# ---------------------------------------------------------------------------

def test_create_action_space_is_binary():
    assert _env().create_action_space().n == 2


# ---------------------------------------------------------------------------
# get_dataloader returns the held loader.
# ---------------------------------------------------------------------------

def test_get_dataloader_returns_attribute():
    e = _env()
    sentinel = object()
    e.dataloader = sentinel
    assert e.get_dataloader() is sentinel


# ---------------------------------------------------------------------------
# render / render_profits are inert no-ops.
# ---------------------------------------------------------------------------

def test_render_methods_return_none():
    e = _env()
    assert e.render() is None
    assert e.render_profits() is None
