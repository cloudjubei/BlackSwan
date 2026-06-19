"""Fast unit tests for ServerDataProvider.prepare_signal.

prepare_signal is the only non-network logic on this class: each incoming signal record is round-tripped
through json -> pandas, fed to the shared AbstractDataProvider.process_df feature pipeline, and its
``.values`` matrix collected. We exercise it with tiny synthetic full-schema frames (no sockets, no
PriceCache, no live server). The live network/socket consumers of this provider are not present in this
module and are intentionally not tested here.

ServerDataProvider does NOT implement the AbstractDataProvider abstractmethods, so it cannot be
instantiated (or even ``__new__``-ed) directly; we subclass it with trivial stubs (the test_abstract
idiom) and bypass __init__ so no data files are touched.
"""

import types

import numpy as np
import pytest

from src.data.server_dataprovider import ServerDataProvider


class _Server(ServerDataProvider):
    """Concrete stub so the ABC can be instantiated; the stubbed methods are never exercised here."""

    def get_timesteps(self):
        return 0

    def get_price(self, step):
        return 0.0

    def get_signal_buy_sell(self, step):
        return 0

    def get_signal_buy_profitable(self, step):
        return 0

    def get_signal_buy_drawdown(self, step):
        return 0

    def get_values(self, step):
        return None


def _server(type="only_price_percent", timestamp="day_of_week", indicator="none"):
    p = _Server.__new__(_Server)
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


def test_cannot_instantiate_directly_abstract_methods_unimplemented():
    # Documents the footgun: the shipped class leaves the 6 abstractmethods unimplemented, so even
    # ``__new__`` raises — it is effectively uninstantiable without a subclass.
    with pytest.raises(TypeError):
        ServerDataProvider.__new__(ServerDataProvider)


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
