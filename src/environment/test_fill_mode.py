"""L6 — trade FILL timing (env_config.fill_mode).

"close" fills at the decision bar's close (same-bar, ~1 bar of look-ahead optimism — the historical
default). "next_open" decides at this bar's close and EXECUTES at the NEXT bar's OPEN, removing that
optimism. Default stays "close" so the existing run corpus + the 27-combo alignment tests are unchanged;
Stage-0 honesty runs opt into "next_open". These tests pin the env fill selection (unit) and the providers'
get_open alignment against the raw open of the same bar get_price's close comes from (data-gated).
"""

import os

import numpy as np
import pytest

import trainer.config_builder as cb
import trainer.walk_forward as wf
from src.conf.env_config import EnvConfig
from src.data.data_factory import create_provider
from src.data.multitimeline_dataprovider import MultiTimelineDataProvider
from src.data.single_dataprovider import SingleDataProvider
from src.environment.trade_all_crypto_env import TradeAllCryptoEnv


class _FakeProvider:
    def __init__(self, closes, opens):
        self._c = [float(x) for x in closes]
        self._o = [float(x) for x in opens]

    def get_timesteps(self):
        return len(self._c) - 1

    def get_lookback_window(self):
        return 1

    def get_price(self, step):
        return self._c[min(max(int(step), 0), len(self._c) - 1)]

    def get_open(self, step):
        return self._o[min(max(int(step), 0), len(self._o) - 1)]

    def get_values(self, step):
        return np.array([0.0], dtype=np.float32)

    def get_signal_buy_sell(self, step):
        return 0

    def get_signal_buy_profitable(self, step):
        return 0

    def get_signal_buy_drawdown(self, step):
        return 0


_CLOSES = [100.0, 110.0, 120.0, 130.0, 140.0]
_OPENS = [100.0, 105.0, 115.0, 125.0, 135.0]  # deliberately distinct from the closes


def _env(fill_mode):
    cfg = EnvConfig(
        type="trade_all", initial_balance=100000, transaction_fee=0.0,
        observations_contain=[], take_profit=None, stop_loss=None, fill_mode=fill_mode,
    )
    env = TradeAllCryptoEnv(cfg, _FakeProvider(_CLOSES, _OPENS), "cpu")
    env.setup("percent_profit", {})
    return env


def test_fill_mode_close_fills_at_decision_bar_close():
    env = _env("close")
    env.current_step = 2
    assert env._fill_price() == _CLOSES[2]  # same-bar close (historical default)


def test_fill_mode_next_open_fills_at_next_bar_open():
    env = _env("next_open")
    env.current_step = 2
    # decide at the close of bar 2 -> EXECUTE at the OPEN of bar 3 (never the same-bar close)
    assert env._fill_price() == _OPENS[3]
    assert env._fill_price() != _CLOSES[2]


def test_fill_mode_next_open_terminal_bar_falls_back_to_close():
    env = _env("next_open")
    env.current_step = env.get_timesteps() - 1  # last decision — no next bar in range
    assert env._fill_price() == env.get_price(env.current_step)


# --- provider get_open alignment (data-gated: needs real klines) -----------------------------------

_HAVE_1D = os.path.exists("binance/BTCUSDT-1d-2020-1.json") and os.path.exists("binance/BTCUSDT-1h-2020-1.json")


def _tiny(cfg=None):
    return [(2020, 1)], [(2020, 2)], {"walk_forward_window": "tiny", "test_to": "2020-02"}


def _build(fidelity_set, monkeypatch):
    monkeypatch.setattr(wf, "resolve_walk_forward_window", _tiny)
    monkeypatch.setattr(cb, "resolve_walk_forward_window", _tiny)
    dc = cb.build_data_config({"timeframe": "1d", "fidelity_set": fidelity_set, "asset": "BTCUSDT"})
    return create_provider(
        dc, dc.train_data_paths, dc.fidelity_input, dc.fidelity_run, list(dc.layers),
        dc.buyreward_maxwait, dc.buyreward_percent,
    )


@pytest.mark.skipif(not _HAVE_1D, reason="needs binance/ BTCUSDT 1d+1h klines")
def test_single_get_open_aligns_with_the_bar_get_price_closes(monkeypatch):
    p = _build("1d", monkeypatch)
    assert isinstance(p, SingleDataProvider)
    raw, *_ = p.get_raw_data(p.paths, p.config.timestamp)
    si = p.get_start_index()
    for step in range(0, p.get_timesteps(), max(1, p.get_timesteps() // 10)):
        assert p.get_price(step) == float(raw["price"].iloc[step + si])  # close alignment
        assert p.get_open(step) == float(raw["price_open"].iloc[step + si])  # open of that SAME bar


@pytest.mark.skipif(not _HAVE_1D, reason="needs binance/ BTCUSDT 1d+1h klines")
def test_multi_get_open_matches_strided_raw_open(monkeypatch):
    p = _build("1h+1d", monkeypatch)
    assert isinstance(p, MultiTimelineDataProvider)
    plot = p.get_raw_df_for_plotting()
    for step in range(0, p.get_timesteps(), max(1, p.get_timesteps() // 10)):
        assert p.get_open(step) == float(plot["price_open"].iloc[step])
