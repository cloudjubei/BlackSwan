"""Fast unit tests for SingleDataProvider.

These bypass the heavy __init__ (which reads JSON files and builds the whole feature frame) via
``Cls.__new__`` and set only the attributes each method reads — synthetic in-memory DataFrames / lists.
The focus is the windowing/lookback grid in get_values and the start-index offsetting that every
accessor shares.
"""

import types

import numpy as np
import pandas as pd
import pytest

from src.data.single_dataprovider import SingleDataProvider


def _provider(df, lookback, prices=None, buy_sell=None, profitable=None, drawdown=None):
    """Build a SingleDataProvider without running __init__; wire only what the method-under-test reads."""
    p = SingleDataProvider.__new__(SingleDataProvider)
    p.config = types.SimpleNamespace(lookback_window_size=lookback)
    p.df = df
    p.prices = prices if prices is not None else list(range(100))
    p.signals_buy_sell = buy_sell if buy_sell is not None else list(range(100))
    p.signals_buy_profitable = profitable if profitable is not None else list(range(100))
    p.signals_buy_drawdown = drawdown if drawdown is not None else list(range(100))
    return p


def _df(n=6, cols=2):
    data = {}
    for c in range(cols):
        data[f"f{c}"] = [float(c * 1000 + i) for i in range(n)]
    return pd.DataFrame(data)


def test_get_start_index_is_lookback_minus_one():
    assert _provider(_df(), lookback=1).get_start_index() == 0
    assert _provider(_df(), lookback=32).get_start_index() == 31


def test_get_timesteps_subtracts_start_index_and_one():
    # df of 6 rows, lookback 3 -> start_index 2 -> 6 - 2 - 1 = 3 usable steps.
    assert _provider(_df(n=6), lookback=3).get_timesteps() == 3


def test_get_timesteps_lookback_one():
    # start_index 0 -> n - 0 - 1.
    assert _provider(_df(n=10), lookback=1).get_timesteps() == 9


def test_get_price_offsets_by_start_index():
    prices = [10.0, 11.0, 12.0, 13.0, 14.0]
    # lookback 3 -> start_index 2, so step 0 reads prices[2].
    p = _provider(_df(), lookback=3, prices=prices)
    assert p.get_price(0) == 12.0
    assert p.get_price(1) == 13.0


def test_get_values_returns_lookback_window_grid():
    # lookback 3 -> start_index 2; step 0 -> offset 2 -> rows [0,1,2] (loc inclusive both ends).
    df = _df(n=6, cols=2)
    p = _provider(df, lookback=3)
    v = p.get_values(0)
    assert v.shape == (3, 2)  # [lookback, per_bar]
    # First column f0 == row index, second column f1 == 1000 + row index.
    assert np.array_equal(v, np.array([[0.0, 1000.0], [1.0, 1001.0], [2.0, 1002.0]]))


def test_get_values_window_slides_with_step():
    df = _df(n=6, cols=1)
    p = _provider(df, lookback=2)  # start_index 1
    # step 1 -> offset 2 -> rows [1,2].
    v = p.get_values(1)
    assert v.shape == (2, 1)
    assert v[:, 0].tolist() == [1.0, 2.0]


def test_get_values_lookback_one_flattens_to_1d():
    df = _df(n=4, cols=3)
    p = _provider(df, lookback=1)  # start_index 0
    v = p.get_values(0)  # offset 0 -> single row [f0,f1,f2] flattened
    assert v.ndim == 1
    assert v.tolist() == [0.0, 1000.0, 2000.0]


def test_get_values_lookback_one_picks_correct_row():
    df = _df(n=5, cols=2)
    p = _provider(df, lookback=1)
    v = p.get_values(3)  # offset 3 -> row 3
    assert v.tolist() == [3.0, 1003.0]


def test_signal_getters_offset_by_start_index():
    bs = [0, 1, 2, 3, 4, 5]
    bp = [10, 11, 12, 13, 14, 15]
    bd = [100, 101, 102, 103, 104, 105]
    p = _provider(_df(), lookback=3, buy_sell=bs, profitable=bp, drawdown=bd)
    # start_index 2 -> step 0 reads index 2 of each list.
    assert p.get_signal_buy_sell(0) == 2
    assert p.get_signal_buy_profitable(0) == 12
    assert p.get_signal_buy_drawdown(0) == 102
    # step 1 -> index 3.
    assert p.get_signal_buy_sell(1) == 3


@pytest.mark.parametrize("lookback", [1, 2, 5])
def test_get_values_row_count_equals_lookback(lookback):
    df = _df(n=12, cols=2)
    p = _provider(df, lookback=lookback)
    v = p.get_values(0)
    # lookback 1 flattens to 1-D (per_bar,), otherwise the first axis is the lookback length.
    if lookback <= 1:
        assert v.ndim == 1
    else:
        assert v.shape[0] == lookback
