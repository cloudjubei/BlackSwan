"""Direct tests for ``ServerCryptoEnv``'s overridden observation builders.

The server env builds a live observation from a fetched values grid. ``__init__`` is bypassed via
``__new__``; collaborators are tiny stubs. ``get_observation`` is a pure list-concatenation helper, so
it is exercised directly with plain Python lists (matching how the provider yields per-bar feature
rows). ``current_step`` is set where the buy path reads it.

Real-flow shapes (see ServerDataProvider.prepare_signal): ``get_values`` yields a 2-D feature grid and
``get_observation`` is meant to append FOUR SCALARS per bar (percent_profit, sl_closeness, drawdown,
position). ``get_observation_buy`` builds those scalars with ``np.zeros_like(values)`` — which only
yields scalars when ``values`` is 1-D, exposing a shape bug for genuine 2-D grids (documented below).
"""

import types

import numpy as np
import pytest

from src.environment.server_crypto_env import ServerCryptoEnv


def _env(values_grid):
    """Build a ServerCryptoEnv whose data_provider returns ``values_grid`` from get_values, and echoes
    its argument from prepare_signal (so the truthiness branch in create_observation is controllable)."""
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
    # With ONE feature per bar, np.zeros_like yields per-bar 1-element arrays that compare equal to 0;
    # the buy-time observation pads every extra field with zeros (no open trade yet).
    e = _env([[5.0], [6.0]])
    out = e.get_observation_buy(values=None)
    # Each bar (one feature) followed by four zeros from the zeros_like padding.
    assert out == [5.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0]


def test_get_observation_buy_ignores_passed_values_and_refetches():
    # CONTRACT NOTE: get_observation_buy takes a `values` arg but immediately overwrites it with a fresh
    # get_values(current_step) fetch, so whatever is passed in is discarded. Pin that behaviour.
    e = _env([[9.0]])
    out_with_garbage = e.get_observation_buy(values=[[1234.0]])
    assert out_with_garbage == [9.0, 0.0, 0.0, 0.0, 0.0]


@pytest.mark.xfail(
    reason="BUG: get_observation_buy uses np.zeros_like(values) so for a multi-FEATURE 2-D grid the "
    "per-bar padding is a vector, not a scalar -> ambiguous truth value / wrong shape",
    strict=False,
)
def test_get_observation_buy_multi_feature_grid_pads_scalars():
    # CONTRACT: each bar should be followed by 4 SCALAR zeros regardless of feature count.
    e = _env([[2.0, 3.0]])
    out = e.get_observation_buy(values=None)
    assert out == [2.0, 3.0, 0.0, 0.0, 0.0, 0.0]


@pytest.mark.xfail(
    reason="BUG: create_observation -> get_observation_buy zeros_like over a multi-feature grid raises",
    strict=False,
)
def test_create_observation_returns_buy_obs_when_signal_truthy_multi_feature():
    e = _env([[2.0, 3.0]])
    out = e.create_observation(signals_data=[[2.0, 3.0]])
    assert out == [2.0, 3.0, 0.0, 0.0, 0.0, 0.0]


def test_create_observation_single_feature_grid_builds_buy_obs():
    # Single-feature grid avoids the zeros_like shape bug, so the happy path is observable.
    e = _env([[2.0]])
    out = e.create_observation(signals_data=[[2.0]])
    assert out == [2.0, 0.0, 0.0, 0.0, 0.0]


def test_create_observation_falsy_signal_returns_none():
    # prepare_signal echoes the (empty) signal -> falsy -> the method falls off the end returning None.
    e = _env([[2.0]])
    assert e.create_observation(signals_data=[]) is None
