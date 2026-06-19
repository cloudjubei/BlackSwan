from src.conf.env_config import (
    EnvConfig,
    get_envs_all,
    get_envs_simple,
    get_envs_swaps,
)


# ---------------------------------------------------------------------------
# EnvConfig dataclass defaults
# ---------------------------------------------------------------------------


def test_env_config_defaults():
    c = EnvConfig()
    assert c.type == "trade_all"
    assert c.initial_balance == 100000
    assert c.amount == 1
    assert c.observations_contain == []
    assert c.transaction_fee == 0.001
    assert c.take_profit is None
    assert c.stop_loss is None
    assert c.no_sell_action is False
    assert c.position_sizing == "fixed"
    assert c.allow_shorting is False
    assert c.batch_size == 32


def test_env_config_observations_contain_factory_is_independent():
    # default_factory=list -> each instance gets its own empty list.
    a = EnvConfig()
    b = EnvConfig()
    assert a.observations_contain is not b.observations_contain


# ---------------------------------------------------------------------------
# get_envs_simple : active selection is env_regression_predict (line 118)
# ---------------------------------------------------------------------------


def test_get_envs_simple_returns_regression_predict():
    envs = get_envs_simple()
    assert isinstance(envs, list)
    assert len(envs) == 1
    assert envs[0].type == "regression_predict"
    assert envs[0].batch_size == 32


# ---------------------------------------------------------------------------
# get_envs_swaps : single swap env (line 136)
# ---------------------------------------------------------------------------


def test_get_envs_swaps_returns_swap_env():
    envs = get_envs_swaps()
    assert isinstance(envs, list)
    assert len(envs) == 1
    assert envs[0].type == "swap"
    assert envs[0].observations_contain == []


# ---------------------------------------------------------------------------
# get_envs_all : the full ordered set (line 139)
# ---------------------------------------------------------------------------


def test_get_envs_all_returns_full_ordered_set():
    envs = get_envs_all()
    assert [e.type for e in envs] == [
        "swap",
        "trade_all",
        "trade_percent",
        "trade_position",
        "trade_amount",
    ]
    # the percent/position/amount variants carry their distinguishing amounts.
    by_type = {e.type: e for e in envs}
    assert by_type["trade_percent"].amount == 0.1
    assert by_type["trade_position"].amount == 100
    assert by_type["trade_amount"].amount == 1
