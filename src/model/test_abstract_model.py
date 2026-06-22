"""Direct tests for the model base-class lifecycle in abstract_model.py.

The classes are bypassed with ``__new__`` (or a tiny concrete subclass where the ABC has abstract
methods) so we exercise the deterministic config-plumbing / id-building / default-flag logic without
constructing a real torch model, data provider, or env. Tiny SimpleNamespace / array stubs stand in
for the env and config collaborators.
"""

import numpy as np
import pytest

from src.conf.model_config import ModelConfig, ModelRLConfig
from src.model.abstract_model import (
    AbstractModel,
    BaseRLModel,
    BaseDeepModel,
    BaseStrategyModel,
)


# --- helpers ---------------------------------------------------------------

def _rl_config(**kw):
    # net_arch / custom_net_arch are pinned for stable ids; the loop env never gets built here so
    # only id-relevant fields matter.
    defaults = dict(
        model_name="dqn",
        reward_model="combo_all",
        net_arch=[512, 64],
        custom_net_arch=[""],
    )
    defaults.update(kw)
    return ModelConfig(model_type="rl", model_rl=ModelRLConfig(**defaults))


class _ConcreteAbstract(AbstractModel):
    """Minimal AbstractModel: only the three abstractmethods, so default flag methods are inherited."""

    def get_id(self, config):
        return "fixed-id"

    def train(self, env):
        return "trained"

    def test(self, env, deterministic=True):
        return "tested"


class _RL(BaseRLModel):
    def train(self, env):
        pass

    def test(self, env, deterministic=True):
        pass


class _Deep(BaseDeepModel):
    def get_id(self, config):
        return "deepid"

    def get_model(self):
        return None

    def get_model_args(self):
        return None

    def get_target_model(self):
        return None

    def get_loss_fn(self):
        return None

    def get_optimizer(self):
        return None

    def get_optimizer_args(self):
        return None


class _Strat(BaseStrategyModel):
    def __init__(self):
        self.config = None

    def get_id(self, config):
        return "sid"

    def get_action(self, env, obs):
        return 1


class _LoopEnv:
    """A tiny env whose step() reports done after n calls; records every action taken."""

    def __init__(self, n, action_value=None):
        self.n = n
        self.i = 0
        self.actions = []
        self._action_value = action_value

    def reset(self):
        self.i = 0
        return (np.zeros(3, dtype=np.float32), {})

    def step(self, action):
        self.actions.append(action)
        self.i += 1
        done = self.i >= self.n
        return (np.zeros(3, dtype=np.float32), 0.0, done, False, {})


# --- AbstractModel defaults ------------------------------------------------

def test_abstract_model_init_stores_config_and_calls_get_id():
    cfg = ModelConfig(model_type="hodl")
    m = _ConcreteAbstract(cfg)
    assert m.config is cfg
    assert m.id == "fixed-id"  # __init__ delegates id-building to get_id


def test_abstract_model_default_flags():
    m = _ConcreteAbstract(ModelConfig(model_type="hodl"))
    assert m.is_pretrained() is False
    assert m.produces_checkpoint() is False
    assert m.show_train_render() is False
    assert m.get_reward_model() == "percent_profit"
    assert m.get_reward_multipliers() == {}
    assert m.has_deterministic_test() is True


# --- BaseRLModel -----------------------------------------------------------

def test_base_rl_model_init_exposes_rl_config():
    cfg = _rl_config()
    m = _RL(cfg)
    assert m.rl_config is cfg.model_rl


def test_base_rl_id_is_sanitised_and_carries_key_fields():
    cfg = _rl_config(learning_rate=0.0001, gamma=0.99)
    m = _RL(cfg)
    # id starts with model_type_model_name_reward_model and dots/pipes are escaped.
    assert m.id.startswith("rl_dqn_combo_all_")
    assert "." not in m.id  # '.' -> '~'
    assert "|" not in m.id  # '|' -> ']'
    assert "0~0001" in m.id  # learning_rate 0.0001 sanitised
    assert "512]64" in m.id  # net_arch [512,64] joined with sanitised pipe


def test_base_rl_id_abbreviates_custom_net_arch():
    # each custom layer name -> first 4 chars + last char, joined with '-'.
    cfg = _rl_config(custom_net_arch=["BatchNorm1d", "Linear"])
    m = _RL(cfg)
    assert "Batcd-Liner" in m.id


def test_base_rl_id_empty_custom_net_arch_produces_no_abbrev():
    # custom_net_arch == [''] is treated as "no custom arch": the abbrev slot is empty.
    cfg = _rl_config(custom_net_arch=[""])
    m = _RL(cfg)
    assert "Batcd" not in m.id


def test_base_rl_is_pretrained_follows_checkpoint_to_load():
    assert _RL(_rl_config(checkpoint_to_load=None)).is_pretrained() is False
    assert _RL(_rl_config(checkpoint_to_load="some/ckpt")).is_pretrained() is True


def test_base_rl_produces_checkpoint_true():
    assert _RL(_rl_config()).produces_checkpoint() is True


def test_base_rl_get_reward_model_uses_config():
    assert _RL(_rl_config(reward_model="profit_all")).get_reward_model() == "profit_all"


def test_base_rl_reward_multipliers_maps_all_combo_keys():
    cfg = _rl_config(
        reward_multiplier_combo_noaction=1.0,
        reward_multiplier_combo_buy=2.0,
        reward_multiplier_combo_fee_penalty=3.0,
        reward_multiplier_combo_noop_penalty=4.0,
    )
    m = _RL(cfg)
    mults = m.get_reward_multipliers()
    assert len(mults) == 19  # 18 combo terms + combo_direct (combo_unified's direct-return weight)
    assert mults["combo_direct"] == cfg.model_rl.reward_multiplier_combo_direct
    assert mults["combo_noaction"] == 1.0
    assert mults["combo_buy"] == 2.0
    assert mults["combo_fee_penalty"] == 3.0
    assert mults["combo_noop_penalty"] == 4.0
    # every value comes straight from the config field of the same name.
    for key, value in mults.items():
        assert getattr(cfg.model_rl, f"reward_multiplier_{key}") == value


def test_base_rl_reward_multiplier_defaults_are_numeric():
    # When the caller leaves the combo_* multipliers at their defaults, the mapped values should be
    # plain numbers usable in reward arithmetic — not (0,) tuples.
    m = _RL(_rl_config())
    mults = m.get_reward_multipliers()
    for key, value in mults.items():
        assert isinstance(value, (int, float)), f"{key} default leaked a non-numeric {value!r}"


# --- BaseDeepModel ---------------------------------------------------------

def test_base_deep_get_episodes_default():
    d = _Deep.__new__(_Deep)
    assert d.get_episodes() == 1


def test_base_deep_checkpoints_path_uses_id():
    d = _Deep.__new__(_Deep)
    d.id = "myid"
    assert d.get_checkpoints_path() == "checkpoints/myid"


def test_base_deep_inherits_abstract_defaults():
    d = _Deep.__new__(_Deep)
    # BaseDeepModel does NOT override produces_checkpoint, even though its train() saves checkpoints.
    assert d.produces_checkpoint() is False
    assert d.is_pretrained() is False
    assert d.get_reward_model() == "percent_profit"
    assert d.get_reward_multipliers() == {}


# --- BaseStrategyModel -----------------------------------------------------

def test_base_strategy_flags():
    s = _Strat()
    assert s.is_pretrained() is True
    assert s.show_train_render() is True


def test_base_strategy_train_steps_until_done_with_get_action():
    s = _Strat()
    env = _LoopEnv(3)
    s.train(env)
    # get_action returns 1 each step; loop runs until step() reports done.
    assert env.actions == [1, 1, 1]


def test_base_strategy_test_delegates_to_train():
    s = _Strat()
    env = _LoopEnv(2)
    s.test(env)
    assert env.actions == [1, 1]
