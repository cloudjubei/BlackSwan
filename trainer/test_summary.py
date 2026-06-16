import pytest

from trainer import summary as summary_mod


class _FakeProvider:
    def __init__(self, prices, lookback=0, timestamps=None):
        self._prices = list(prices)
        self._lookback = lookback
        self.timestamps = list(timestamps) if timestamps is not None else []

    def get_price(self, i):
        return self._prices[i]

    def get_lookback_window(self):
        return self._lookback


class _FakeEnv:
    def __init__(self, net_worths, actions, prices, tpsls=None, lookback=0, timestamps=None):
        self.net_worths = list(net_worths)
        self.actions = list(actions)
        self.tpsls = list(tpsls) if tpsls is not None else [0] * len(actions)
        self.data_provider = _FakeProvider(prices, lookback, timestamps)


class _FakeModel:
    id = None


def _state(n_trades=25, win=50.0):
    s = [0.0] * 19
    s[7] = win
    s[17] = n_trades
    return s


def _build(cfg, net_worths=None, actions=None, prices=None):
    if net_worths is None:
        net_worths = [100000, 101000, 100500, 102000, 101800, 103000]
    if actions is None:
        actions = [0, 1, 0, 2, 1, 0]
    if prices is None:
        prices = [100, 110, 105, 120, 115, 130]
    env = _FakeEnv(net_worths, actions, prices)
    return summary_mod.build_summary(env, _state(), cfg, _FakeModel(), "2026-01-01T00:00:00Z", True)


def test_sharpe_alpha_is_strategy_minus_hold_sharpe():
    out = _build({"timeframe": "1d", "walk_forward_window": "2022", "lookback_window_size": 0})
    assert "sharpe_alpha" in out["metrics"]
    assert out["metrics"]["sharpe_alpha"] == pytest.approx(
        out["metrics"]["sharpe"] - out["benchmark"]["hold_sharpe"]
    )


def test_dataset_stamps_the_walk_forward_window():
    out = _build({"timeframe": "1d", "walk_forward_window": "2023", "lookback_window_size": 0})
    assert out["dataset"]["walk_forward_window"] == "2023"


def test_dataset_window_defaults_to_2024_when_absent():
    out = _build({"timeframe": "1d", "lookback_window_size": 0})
    assert out["dataset"]["walk_forward_window"] == "2024"


def test_sharpe_alpha_absent_without_a_benchmark():
    out = _build(
        {"timeframe": "1d", "lookback_window_size": 0},
        net_worths=[100000, 101000],
        actions=[1],
        prices=[100],
    )
    assert out.get("benchmark") is None
    assert "sharpe_alpha" not in out["metrics"]
