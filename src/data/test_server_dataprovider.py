"""Fast unit tests for ServerDataProvider.

ServerDataProvider is the live-inference provider: it has no data files, instead each incoming signal
record is round-tripped through json -> pandas, fed to the shared AbstractDataProvider.process_df feature
pipeline, and its ``.values`` matrix collected by prepare_signal (which also caches the latest prices /
timestamps as instance state). We exercise it with tiny synthetic full-schema frames (no sockets, no
PriceCache, no live server). The live network/socket consumers of this provider are not present in this
module and are intentionally not tested here.

ServerDataProvider is a CONCRETE provider: it implements every AbstractDataProvider abstractmethod, so it
instantiates directly. The forward-looking signal_* getters have no future labels under live inference and
return a neutral 0; get_values/get_price/get_timesteps derive from the latest prepared signal.
"""

import types

import numpy as np
import pytest

from src.data.server_dataprovider import ServerDataProvider


def _server(type="only_price_percent", timestamp="day_of_week", indicator="none"):
    p = ServerDataProvider.__new__(ServerDataProvider)
    # process_df reads only these config fields.
    p.config = types.SimpleNamespace(
        type=type,
        timestamp=timestamp,
        indicator=indicator,
        buyreward_percent=0.004,
        buyreward_maxwait=20,
        use_indicators=False,
        obs_squash="none",
    )
    # __init__ initialises these state holders; __new__ bypasses it, so seed them here.
    p.values = []
    p.prices = []
    p.timestamps = []
    return p


def _signal(n):
    """A full-schema kline record as a dict-of-lists (survives json.dumps -> pd.read_json into columns)."""
    base = 1_600_000_000_000
    price = [100.0 * (1.01**i) for i in range(n)]
    return {
        "timestamp": [base + i * 60000 for i in range(n)],
        "timestamp_close": [base + i * 60000 for i in range(n)],
        "price": price,
        "price_open": price,
        "price_high": [p * 1.01 for p in price],
        "price_low": [p * 0.99 for p in price],
        "volume": [1000.0 + i for i in range(n)],
        "asset_volume_quote": [5000.0 + i for i in range(n)],
        "trades_number": [10 + i for i in range(n)],
        "asset_volume_taker_base": [500.0 + i for i in range(n)],
        "asset_volume_taker_quote": [2500.0 + i for i in range(n)],
    }


def test_is_concrete_and_instantiates_directly():
    # ServerDataProvider implements all 6 AbstractDataProvider abstractmethods, so it is concrete and
    # constructs directly (no stub subclass needed). __init__ seeds empty state holders.
    config = types.SimpleNamespace(
        id="latest",
        type="only_price_percent",
        timestamp="day_of_week",
        indicator="none",
        fidelity_input="1m",
        fidelity_run="1m",
        fidelity_input_test="1m",
        fidelity_run_test="1m",
        layers=["1h", "1d"],
        layers_test=["1h", "1d"],
        lookback_window_size=1,
        buyreward_percent=0.004,
        buyreward_maxwait=20,
        use_indicators=False,
        obs_squash="none",
    )
    p = ServerDataProvider(config)
    assert isinstance(p, ServerDataProvider)
    # Before any signal has been prepared the provider is empty / neutral, never raising.
    assert p.get_timesteps() == 0
    assert p.get_values(0) == []
    assert p.get_signal_buy_sell(0) == 0
    assert p.get_signal_buy_profitable(0) == 0
    assert p.get_signal_buy_drawdown(0) == 0


def test_get_values_returns_prepared_value_matrices():
    p = _server()
    out = p.prepare_signal([_signal(5), _signal(6)])
    # get_values exposes exactly what prepare_signal cached (the list of per-signal value matrices).
    got = p.get_values(0)
    assert len(got) == 2
    assert got is p.values
    for a, b in zip(got, out):
        assert np.array_equal(a, b)


def test_get_timesteps_counts_prepared_signals():
    p = _server()
    p.prepare_signal([_signal(5), _signal(6), _signal(4)])
    assert p.get_timesteps() == 3


def test_get_price_is_latest_price_of_signal_window():
    p = _server()
    p.prepare_signal([_signal(5), _signal(6)])
    # process_df returns the raw prices array per signal; get_price reads its last (most recent) entry.
    expected_0 = 100.0 * (1.01**4)
    expected_1 = 100.0 * (1.01**5)
    assert p.get_price(0) == pytest.approx(expected_0)
    assert p.get_price(1) == pytest.approx(expected_1)


def test_signal_getters_return_neutral_zero():
    # Live inference has no future labels, so the forward-looking signal_* getters default to 0.
    p = _server()
    p.prepare_signal([_signal(5)])
    assert p.get_signal_buy_sell(0) == 0
    assert p.get_signal_buy_profitable(0) == 0
    assert p.get_signal_buy_drawdown(0) == 0


def test_prepare_signal_one_array_per_input_signal():
    p = _server()
    out = p.prepare_signal([_signal(5), _signal(6), _signal(4)])
    assert len(out) == 3


def test_prepare_signal_preserves_row_count_per_signal():
    p = _server()
    out = p.prepare_signal([_signal(5), _signal(7)])
    # process_df keeps every input row (no rows dropped), so the value matrix has the same length.
    assert out[0].shape[0] == 5
    assert out[1].shape[0] == 7


def test_prepare_signal_returns_numpy_value_matrices():
    p = _server()
    out = p.prepare_signal([_signal(5)])
    assert isinstance(out[0], np.ndarray)
    assert out[0].ndim == 2  # [rows, features]


def test_prepare_signal_empty_list_returns_empty():
    p = _server()
    assert p.prepare_signal([]) == []


def test_prepare_signal_feature_width_consistent_across_signals():
    # Same config -> same feature engineering -> identical column count regardless of row count.
    p = _server()
    out = p.prepare_signal([_signal(5), _signal(9)])
    assert out[0].shape[1] == out[1].shape[1]
