import glob
import types

import numpy as np
import pandas as pd
import pytest

from src.conf.data_config import DataConfig
from src.data import abstract_dataprovider as adp
from src.data.abstract_dataprovider import AbstractDataProvider
from src.data.single_dataprovider import SingleDataProvider
from trainer.pit_fusion import release_datetime_ms

_KLINE_COLUMNS = [
    "timestamp", "timestamp_close", "price", "price_open", "price_high", "price_low",
    "volume", "asset_volume_quote", "trades_number", "asset_volume_taker_base",
]


def _provider(context):
    return types.SimpleNamespace(config=types.SimpleNamespace(context=context))


def test_context_none_is_a_no_op():
    df = pd.DataFrame({"price_percent": [0.1, 0.2]})
    out = AbstractDataProvider._add_context_columns(_provider("none"), df, np.array([1, 2]))
    assert list(out.columns) == ["price_percent"]


def test_context_none_keeps_the_existing_cache_key():
    # Non-context runs must keep their exact feature-cache key (no mass rebuild); context runs get one.
    assert AbstractDataProvider._context_cache_key(_provider("none")) is None
    assert AbstractDataProvider._context_cache_key(_provider("rates")) == "rates"


def test_context_panel_injects_leakage_safe_level_columns(monkeypatch):
    r = release_datetime_ms("2020-01-10", "16:00")  # DFF publishes 16:00 ET
    ts = np.array([r - 1, r, r + 1])
    df = pd.DataFrame({"price_percent": [0.0, 0.0, 0.0]})
    monkeypatch.setattr(
        adp, "load_series_observations",
        lambda sid, source: [{"releaseDate": "2020-01-10", "value": 4.0}],
    )
    out = AbstractDataProvider._add_context_columns(_provider("rates"), df, ts)
    # rates = DFF, DGS10, DFII10, T10Y2Y -> four raw-level context columns, in panel order.
    assert [c for c in out.columns if c.startswith("context_")] == [
        "context_DFF", "context_DGS10", "context_DFII10", "context_T10Y2Y",
    ]
    dff = out["context_DFF"].tolist()  # DFF level x scale 0.1 = 0.4; None (NaN) before the 16:00 release
    assert np.isnan(dff[0])
    assert dff[1] == pytest.approx(0.4)
    assert dff[2] == pytest.approx(0.4)


def test_price_source_context_flows_into_a_real_observation():
    # LIVE end-to-end on data ALREADY on disk (no external mine): an alt's real klines run through the real
    # process_df_simple with the `majors` panel -> BTC/ETH daily returns fused as leakage-safe context.
    ada = sorted(glob.glob("binance/ADAUSDT-1d-*.json"))
    if not (ada and glob.glob("binance/BTCUSDT-1d-*.json") and glob.glob("binance/ETHUSDT-1d-*.json")):
        pytest.skip("needs ADA/BTC/ETH daily klines on disk")
    raw = pd.concat([pd.read_json(p) for p in ada[:2]], ignore_index=True)[_KLINE_COLUMNS]

    provider = SingleDataProvider.__new__(SingleDataProvider)
    provider.config = DataConfig(id="t", train_data_paths=[], test_data_paths=[], context="majors")
    out, prices, timestamps = provider.process_df_simple(raw.copy(), "none", list(_KLINE_COLUMNS))

    assert "context_BTCUSDT" in out.columns and "context_ETHUSDT" in out.columns
    # Real returns actually fused in (not a dead zero column) — proves the bar-clock matched the release ms.
    assert out["context_BTCUSDT"].abs().sum() > 0
    assert out["context_ETHUSDT"].abs().sum() > 0
