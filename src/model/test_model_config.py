"""Direct tests for the model config dataclasses in conf/model_config.py.

These guard the dataclass DEFAULTS: every list field must use ``default_factory=list`` (or a
``lambda`` for seeded lists) so the config constructs without TypeError, and the scalar
reward-multiplier defaults must be plain numbers — not the ``(0,)`` tuples that trailing commas
used to produce.
"""

from src.conf.model_config import (
    ModelRegressionConfig,
    ModelRegressionConfigSearch,
    ModelRLConfig,
    ModelRLConfigSearch,
    ModelTimeConfigSearch,
)


# --- list defaults construct cleanly ---------------------------------------

def test_rl_config_net_arch_defaults_to_empty_list():
    cfg = ModelRLConfig(model_name="dqn", reward_model="combo_all")
    assert cfg.net_arch == []
    assert cfg.custom_net_arch == []
    assert isinstance(cfg.net_arch, list)
    assert isinstance(cfg.custom_net_arch, list)


def test_rl_config_default_lists_are_independent_instances():
    # default_factory=list must give each instance its own list, not a shared mutable default.
    a = ModelRLConfig(model_name="dqn", reward_model="combo_all")
    b = ModelRLConfig(model_name="dqn", reward_model="combo_all")
    a.net_arch.append(1)
    assert b.net_arch == []


def test_regression_config_constructs_without_args():
    # Previously raised TypeError because default_factory=[] is not callable.
    cfg = ModelRegressionConfig()
    assert cfg.net_arch == []
    assert cfg.custom_net_arch == []


def test_rl_config_search_constructs_with_list_defaults():
    cfg = ModelRLConfigSearch()
    assert cfg.model_name == []
    assert cfg.reward_model == []
    assert cfg.net_arch == []
    assert cfg.reward_multiplier_combo_buy == []


def test_regression_config_search_constructs_with_list_defaults():
    cfg = ModelRegressionConfigSearch()
    assert cfg.model_name == []
    assert cfg.net_arch == []


def test_time_config_search_seeded_lists_are_preserved():
    # Seeded list literals must keep their values via a lambda factory.
    cfg = ModelTimeConfigSearch()
    assert cfg.time_buy == [1200]
    assert cfg.time_sell == [1400]


# --- reward-multiplier scalar defaults -------------------------------------

def test_rl_config_reward_multiplier_defaults_are_scalars():
    # Trailing commas used to make these the tuple (0,); they must be plain numbers.
    cfg = ModelRLConfig(model_name="dqn", reward_model="combo_all")
    multiplier_fields = [
        "reward_multiplier_combo_sell_profit",
        "reward_multiplier_combo_sell_profit_prev",
        "reward_multiplier_combo_sell_perfect",
        "reward_multiplier_combo_sell_drawdown",
        "reward_multiplier_combo_buy_profit",
        "reward_multiplier_combo_buy_perfect",
        "reward_multiplier_combo_buy_profitable_offset",
        "reward_multiplier_combo_buy_profitable",
        "reward_multiplier_combo_buy_drawdown",
        "reward_multiplier_combo_hold_profit",
        "reward_multiplier_combo_hold_drawdown",
    ]
    for name in multiplier_fields:
        value = getattr(cfg, name)
        assert isinstance(value, (int, float)), f"{name} default leaked a non-numeric {value!r}"
        assert value == 0
