"""Direct tests for ``ServerCryptoEnv``'s overridden observation builders.

The server env builds a live observation from a fetched values grid. ``__init__`` is bypassed via
``__new__``; collaborators are tiny stubs. ``get_observation`` is a pure list-concatenation helper, so
it is exercised directly with plain Python lists (matching how the provider yields per-bar feature
rows). ``current_step`` is set where the buy path reads it.

Real-flow shapes (see ServerDataProvider.prepare_signal): ``prepare_signal`` yields a 2-D feature grid
that flows into ``get_observation_buy`` AS the live ``values``, and ``get_observation`` appends FOUR
SCALARS per bar (percent_profit, sl_closeness, drawdown, position). ``get_observation_buy`` builds those
scalars from a per-bar scalar zero array, so padding works for genuine multi-feature 2-D grids.
"""

import types

import numpy as np
import pytest

from src.environment.server_crypto_env import ServerCryptoEnv


def _env(values_grid):
    """Build a ServerCryptoEnv whose data_provider echoes its argument from prepare_signal (so the
    truthiness branch in create_observation is controllable and the prepared grid flows into the buy
    observation). ``get_values`` returns ``values_grid`` and must be IGNORED by the buy path."""
    e = ServerCryptoEnv.__new__(ServerCryptoEnv)
    e.current_step = 0
    e.data_provider = types.SimpleNamespace(
        get_values=lambda step: values_grid,
        prepare_signal=lambda signals: signals,
    )
    return e


def test_get_observation_interleaves_per_bar_extra_fields():
    # Two bars, each a 2-feature list; for each bar append [profit, sl_closeness, drawdown, position].
    # The extra fields are passed as 1-D per-bar SCALAR arrays, matching the documented signature.
    e = ServerCryptoEnv.__new__(ServerCryptoEnv)
    values = [[1.0, 2.0], [3.0, 4.0]]
    out = e.get_observation(
        values,
        percent_profits=[0.1, 0.2],
        stoploss_closeness=[0.0, 0.0],
        drawdowns=[-0.05, -0.06],
        positions=[1, 1],
    )
    # bar0: 1,2 + [0.1,0,-0.05,1] ; bar1: 3,4 + [0.2,0,-0.06,1]
    assert out == [1.0, 2.0, 0.1, 0.0, -0.05, 1, 3.0, 4.0, 0.2, 0.0, -0.06, 1]


def test_get_observation_empty_values_returns_empty_list():
    e = ServerCryptoEnv.__new__(ServerCryptoEnv)
    assert e.get_observation([], [], [], [], []) == []


def test_get_observation_requires_list_rows_for_concatenation():
    # out = out + values[i] + [...] is LIST concatenation: a bar must be a list, not a scalar.
    e = ServerCryptoEnv.__new__(ServerCryptoEnv)
    out = e.get_observation(
        [[5.0]],
        percent_profits=[0.0],
        stoploss_closeness=[0.0],
        drawdowns=[0.0],
        positions=[0.0],
    )
    assert out == [5.0, 0.0, 0.0, 0.0, 0.0]


def test_get_observation_buy_zeroes_all_extra_fields_for_single_feature_grid():
    # With ONE feature per bar, the buy-time observation pads every extra field with scalar zeros
    # (no open trade yet). The grid is consumed from the passed-in `values`, NOT refetched.
    e = _env(None)
    out = e.get_observation_buy(values=[[5.0], [6.0]])
    # Each bar (one feature) followed by four scalar zeros from the padding.
    assert out == [5.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0]


def test_get_observation_buy_uses_passed_values():
    # CONTRACT: get_observation_buy builds the observation from the `values` it is given (the live
    # signal grid prepared by create_observation), it does NOT discard them and refetch get_values.
    e = _env([[9999.0]])  # get_values would return garbage; it must be ignored
    out = e.get_observation_buy(values=[[42.0]])
    assert out == [42.0, 0.0, 0.0, 0.0, 0.0]


def test_get_observation_buy_multi_feature_grid_pads_scalars():
    # Each bar is followed by 4 SCALAR zeros regardless of feature count — works for a real 2-D
    # numpy feature grid (multiple features per bar).
    e = _env(None)
    values = np.array([[2.0, 3.0]])
    out = e.get_observation_buy(values=values)
    assert out == [2.0, 3.0, 0.0, 0.0, 0.0, 0.0]


def test_create_observation_returns_buy_obs_when_signal_truthy_multi_feature():
    # The live multi-feature grid from prepare_signal (per-bar list rows) flows straight into the buy
    # observation; each bar is padded with four scalar zeros regardless of feature count.
    e = _env(None)
    out = e.create_observation(signals_data=[[2.0, 3.0]])
    assert out == [2.0, 3.0, 0.0, 0.0, 0.0, 0.0]


def test_create_observation_single_feature_grid_builds_buy_obs():
    e = _env(None)
    out = e.create_observation(signals_data=[[2.0]])
    assert out == [2.0, 0.0, 0.0, 0.0, 0.0]


def test_create_observation_falsy_signal_returns_none():
    # prepare_signal echoes the (empty) signal -> falsy -> the method falls off the end returning None.
    e = _env([[2.0]])
    assert e.create_observation(signals_data=[]) is None
