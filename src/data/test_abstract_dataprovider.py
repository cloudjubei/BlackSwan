import numpy as np
import pandas as pd

from src.conf.data_config import DataConfig
from src.data.abstract_dataprovider import AbstractDataProvider

_COLUMNS = [
    "timestamp",
    "timestamp_close",
    "price",
    "price_open",
    "price_high",
    "price_low",
    "volume",
    "asset_volume_quote",
    "trades_number",
    "asset_volume_taker_base",
]


class _Provider(AbstractDataProvider):
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


def _provider(**cfg_kw):
    cfg = DataConfig(
        id="t",
        train_data_paths=[[]],
        test_data_paths=[[]],
        layers=["1d"],
        layers_test=["1d"],
        **cfg_kw,
    )
    return _Provider(cfg)


def _df(n=40):
    base = 1_600_000_000_000
    price = [100.0 * (1.02**i) for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": [base + i * 86400000 for i in range(n)],
            "timestamp_close": [base + i * 86400000 for i in range(n)],
            "price": price,
            "price_open": price,
            "price_high": [p * 1.01 for p in price],
            "price_low": [p * 0.99 for p in price],
            "volume": [1000.0 + i for i in range(n)],
            "asset_volume_quote": [50000.0 + i for i in range(n)],
            "trades_number": [10 + i for i in range(n)],
            "asset_volume_taker_base": [500.0 + i for i in range(n)],
        }
    )


def test_squash_features_clip():
    p = _provider(obs_squash="clip")
    out = p._squash_observation_features(pd.DataFrame({"a": [2.0, -3.0, 0.5], "b": [1.5, 0.0, -0.2]}))
    assert out["a"].tolist() == [1.0, -1.0, 0.5]
    assert out["b"].tolist() == [1.0, 0.0, -0.2]


def test_squash_features_tanh_bounds_everything():
    p = _provider(obs_squash="tanh")
    out = p._squash_observation_features(pd.DataFrame({"a": [5.0, -5.0, 0.0]}))
    assert out["a"].abs().le(1.0).all()
    assert out["a"].iloc[2] == 0.0


def test_squash_features_none_is_identity():
    p = _provider(obs_squash="none")
    out = p._squash_observation_features(pd.DataFrame({"a": [2.0, -3.0]}))
    assert out["a"].tolist() == [2.0, -3.0]


def test_curated_indicators_add_regime_and_trend_features():
    out = _provider(use_indicators=True)._add_curated_indicators(_df())
    assert "volRegime10" in out.columns
    assert "trendSlope10" in out.columns
    vals = pd.concat([out["volRegime10"], out["trendSlope10"]]).dropna()
    assert vals.between(-1.0, 1.0).all()


def test_curated_indicators_skipped_without_flag():
    out = _provider(use_indicators=False)._add_curated_indicators(_df())
    assert "volRegime10" not in out.columns


def test_process_df_simple_runs_and_clip_bounds_output():
    p = _provider(obs_squash="clip")
    out, prices, timestamps = p.process_df_simple(_df(), "day_of_week", list(_COLUMNS))
    nums = out.select_dtypes(include=[np.number])
    assert (nums.abs() <= 1.0 + 1e-9).all().all()
    assert len(prices) == 40
