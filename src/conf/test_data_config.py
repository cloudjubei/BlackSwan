import pytest

from src.conf.data_config import (
    DataConfig,
    get_datas_1h_1d,
    get_datas_1m_1h_1d,
)


# ---------------------------------------------------------------------------
# DataConfig dataclass defaults (layers must be supplied; see xfail below)
# ---------------------------------------------------------------------------


def test_data_config_defaults():
    # layers / layers_test cannot be defaulted (broken factory), so they are passed explicitly here.
    c = DataConfig(
        id="t", train_data_paths=[[]], test_data_paths=[[]], layers=["1d"], layers_test=["1d"]
    )
    assert c.lookback_window_size == 1
    assert c.type == "only_price_percent"
    assert c.use_indicators is False
    assert c.timestamp == "day_of_week"
    assert c.buyreward_percent == 0.004
    assert c.buyreward_maxwait == 20
    assert c.fidelity_input == "1m"
    assert c.fidelity_run == "1m"
    assert c.obs_squash == "none"


@pytest.mark.xfail(
    reason="BUG: layers/layers_test use field(default_factory=[]); an empty-list literal is not "
    "callable, so constructing DataConfig without those args raises 'list object is not callable'. "
    "Should be default_factory=list.",
    strict=False,
    raises=TypeError,
)
def test_data_config_constructs_with_default_layers():
    # Intended contract: layers/layers_test default to empty lists.
    c = DataConfig(id="t", train_data_paths=[[]], test_data_paths=[[]])
    assert c.layers == []
    assert c.layers_test == []


# ---------------------------------------------------------------------------
# get_datas_1h_1d : currently returns the at_5m_buyreward variant (line 216)
# ---------------------------------------------------------------------------


def test_get_datas_1h_1d_returns_single_5m_buyreward_config():
    datas = get_datas_1h_1d()
    assert isinstance(datas, list)
    assert len(datas) == 1
    cfg = datas[0]
    assert isinstance(cfg, DataConfig)
    # the active selection is the *_at_5m_buyreward variant.
    assert cfg.fidelity_run == "5m"
    assert cfg.lookback_window_size == 32
    assert cfg.layers == ["5m", "15m", "1h", "4h", "1d"]
    assert cfg.buyreward_percent == 0.003
    assert cfg.buyreward_maxwait == 6
    assert cfg.buyreward_maxwait_test == 6 * 5


# ---------------------------------------------------------------------------
# get_datas_1m_1h_1d : returns the 1m-layered variant (line 228)
# ---------------------------------------------------------------------------


def test_get_datas_1m_1h_1d_returns_single_1m_config():
    datas = get_datas_1m_1h_1d()
    assert isinstance(datas, list)
    assert len(datas) == 1
    cfg = datas[0]
    assert cfg.fidelity_run == "1m"
    assert cfg.layers == ["1m", "1h", "1d"]
    assert cfg.layers_test == ["1m", "1h", "1d"]
    assert cfg.lookback_window_size == 32
