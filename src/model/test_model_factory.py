"""Unit tests for the model construction / dispatch hub.

These exercise the DETERMINISTIC dispatch + config-mapping logic only: which model class /
policy / network a given model_type / model_name selects, and that hyperparameters thread
through to the constructor. Heavy sb3 / torch construction is avoided by monkeypatching the
algorithm constructors in the module namespace and asserting the captured kwargs (the smoke
test that actually builds a PPO lives in custom/test_sequence_extractor.py).
"""

import types

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.conf.model_config import (
    ModelConfig,
    ModelConfigSearch,
    ModelRegressionConfig,
    ModelRLConfig,
    ModelTechnicalConfig,
    ModelTechnicalConfigSearch,
    ModelTimeConfig,
    ModelTimeConfigSearch,
    ModelMomentumConfig,
    ModelMomentumConfigSearch,
    ModelSupervisedConfig,
)
from src.model import model_factory
from src.model.model_factory import (
    create_model,
    create_regression_model,
    create_rl_model,
    get_combo,
    get_model_combinations,
    _resolve_pos_weight,
)
from src.model.custom.policies import (
    CustomActorCriticPolicy,
    CustomDQNPolicy,
    CustomDuelingDQNPolicy,
    CustomQRDQNPolicy,
    CustomRecurrentActorCriticPolicy,
)
from src.model.custom.policy_iqn import CustomIQNPolicy
from src.model.hodl_model import HodlModel
from src.model.rl_model import RLModel
from src.model.regression_model import RegressionModel
from src.model.supervised_model import SupervisedModel
from src.model.technical_strategy_model import TechnicalStrategyModel
from src.model.time_strategy_model import TimeStrategyModel
from src.model.momentum_strategy_model import MomentumStrategyModel


# --------------------------------------------------------------------------------------
# get_combo — pure dict assembly
# --------------------------------------------------------------------------------------

def test_get_combo_zips_keys_to_values_and_merges_non_lists():
    out = get_combo(("a", "b"), ["x", "y"], {"z": 1})
    assert out == {"x": "a", "y": "b", "z": 1}


def test_get_combo_non_list_values_override_zipped_on_key_clash():
    # update() applies the non-list values last, so a shared key takes the non-list value.
    out = get_combo(("first",), ["k"], {"k": "override"})
    assert out == {"k": "override"}


def test_get_combo_empty_inputs_give_only_non_lists():
    assert get_combo((), [], {"only": 9}) == {"only": 9}


# --------------------------------------------------------------------------------------
# get_model_combinations — dispatch by model_type + cartesian expansion
# --------------------------------------------------------------------------------------

def test_get_model_combinations_hodl_is_single_config():
    combos = get_model_combinations(OmegaConf.structured(ModelConfigSearch(model_type="hodl")))
    assert len(combos) == 1
    assert combos[0].model_type == "hodl"


def test_get_model_combinations_time_cartesian_product():
    search = OmegaConf.structured(
        ModelConfigSearch(
            model_type="time",
            model_time=ModelTimeConfigSearch(time_buy=[1200, 1300], time_sell=[1400]),
        )
    )
    combos = get_model_combinations(search)
    # 2 buys x 1 sell = 2 fully-resolved configs.
    assert len(combos) == 2
    assert [c.model_type for c in combos] == ["time", "time"]
    assert {(c.model_time.time_buy, c.model_time.time_sell) for c in combos} == {
        (1200, 1400),
        (1300, 1400),
    }


def test_get_model_combinations_technical_full_cartesian():
    search = OmegaConf.structured(
        ModelConfigSearch(
            model_type="technical",
            model_technical=ModelTechnicalConfigSearch(
                buy_indicator=["rsi5", "rsi10"],
                buy_amount_threshold=[30.0],
                sell_indicator=["rsi5"],
                sell_amount_threshold=[70.0, 80.0],
            ),
        )
    )
    combos = get_model_combinations(search)
    # 2 buy_indicator x 1 x 1 x 2 sell_amount_threshold = 4
    assert len(combos) == 4
    assert all(c.model_type == "technical" for c in combos)
    pairs = {(c.model_technical.buy_indicator, c.model_technical.sell_amount_threshold) for c in combos}
    assert pairs == {("rsi5", 70.0), ("rsi5", 80.0), ("rsi10", 70.0), ("rsi10", 80.0)}


def test_get_model_combinations_non_list_values_stay_scalar():
    # buy_amount_is_multiplier is a plain bool (not a ListConfig) so it is carried verbatim.
    search = OmegaConf.structured(
        ModelConfigSearch(
            model_type="technical",
            model_technical=ModelTechnicalConfigSearch(
                buy_indicator=["rsi5"],
                buy_amount_threshold=[30.0],
                sell_indicator=["rsi5"],
                sell_amount_threshold=[70.0],
                buy_amount_is_multiplier=True,
            ),
        )
    )
    combos = get_model_combinations(search)
    assert len(combos) == 1
    assert combos[0].model_technical.buy_amount_is_multiplier is True


def test_get_model_combinations_momentum_cartesian_product():
    search = OmegaConf.structured(
        ModelConfigSearch(
            model_type="momentum",
            model_momentum=ModelMomentumConfigSearch(lookback_periods=[30, 90, 252]),
        )
    )
    combos = get_model_combinations(search)
    assert [c.model_type for c in combos] == ["momentum", "momentum", "momentum"]
    assert {c.model_momentum.lookback_periods for c in combos} == {30, 90, 252}


def test_get_model_combinations_unsupported_type_raises():
    with pytest.raises(ValueError):
        get_model_combinations(OmegaConf.structured(ModelConfigSearch(model_type="nope")))


# --------------------------------------------------------------------------------------
# create_model — dispatch by model_type (cheap, non-RL branches build for real)
# --------------------------------------------------------------------------------------

def test_create_model_hodl_returns_hodl_model():
    m = create_model(ModelConfig(model_type="hodl"), env=None, device="cpu")
    assert isinstance(m, HodlModel)


def test_create_model_time_returns_time_strategy():
    cfg = ModelConfig(model_type="time", model_time=ModelTimeConfig(time_buy=1200, time_sell=1400))
    m = create_model(cfg, env=None, device="cpu")
    assert isinstance(m, TimeStrategyModel)


def test_create_model_technical_returns_technical_strategy():
    cfg = ModelConfig(
        model_type="technical",
        model_technical=ModelTechnicalConfig(
            buy_indicator="rsi5", buy_amount_threshold=30.0, sell_indicator="rsi10", sell_amount_threshold=70.0
        ),
    )
    m = create_model(cfg, env=None, device="cpu")
    assert isinstance(m, TechnicalStrategyModel)


def test_create_model_supervised_returns_supervised_model():
    cfg = ModelConfig(model_type="supervised", model_supervised=ModelSupervisedConfig())
    m = create_model(cfg, env=None, device="cpu")
    assert isinstance(m, SupervisedModel)


def test_create_model_momentum_returns_momentum_strategy():
    cfg = ModelConfig(model_type="momentum", model_momentum=ModelMomentumConfig(lookback_periods=30))
    m = create_model(cfg, env=None, device="cpu")
    assert isinstance(m, MomentumStrategyModel)
    assert m.momentum_config.lookback_periods == 30


def test_create_model_rl_delegates_to_create_rl_model(monkeypatch):
    sentinel = object()
    seen = {}

    def fake_create_rl(config, env, device):
        seen["args"] = (config, env, device)
        return sentinel

    monkeypatch.setattr(model_factory, "create_rl_model", fake_create_rl)
    cfg = ModelConfig(model_type="rl", model_rl=_rl_config("ppo"))
    out = create_model(cfg, env="ENV", device="cuda")
    assert out is sentinel
    assert seen["args"] == (cfg, "ENV", "cuda")


def test_create_model_regression_delegates_to_create_regression_model(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(model_factory, "create_regression_model", lambda c, e, d: sentinel)
    # net_arch / custom_net_arch passed explicitly to dodge the dataclass default_factory bug
    # (see test file notes) — this test only exercises create_model's regression dispatch.
    cfg = ModelConfig(
        model_type="regression",
        model_regression=ModelRegressionConfig(net_arch=[4], custom_net_arch=[]),
    )
    assert create_model(cfg, env="ENV", device="cpu") is sentinel


def test_create_model_unsupported_type_raises():
    with pytest.raises(ValueError):
        create_model(ModelConfig(model_type="bogus"), env=None, device="cpu")


# --------------------------------------------------------------------------------------
# create_rl_model — dispatch + hyperparameter mapping (monkeypatched constructors)
# --------------------------------------------------------------------------------------

def _rl_config(model_name, **overrides):
    base = dict(
        model_name=model_name,
        reward_model="combo_all",
        net_arch=[64, 32],
        custom_net_arch=[],
        optimizer_class="Adam",
        activation_fn="ReLU",
        learning_rate=0.0003,
        batch_size=16,
        gamma=0.97,
        seed=None,
    )
    base.update(overrides)
    return ModelRLConfig(**base)


class _FakeAlgo:
    """Captures the constructor kwargs and records the post-construction calls."""

    last = {}

    def __init__(self, **kwargs):
        type(self).last = dict(kwargs)
        self.set_logger_called = False
        self.set_seed = None

    def set_logger(self, logger):
        self.set_logger_called = True

    def set_random_seed(self, seed):
        self.set_seed = seed


def _patch_algo(monkeypatch, attr):
    """Patch one algorithm constructor in the factory namespace with a fresh fake class."""

    class Algo(_FakeAlgo):
        pass

    monkeypatch.setattr(model_factory, attr, Algo)
    return Algo


# (model_name, factory-namespace attr to patch, expected policy value)
_DISPATCH_CASES = [
    ("ppo", "PPO", "MlpPolicy"),
    ("ppo-custom", "PPO", CustomActorCriticPolicy),
    ("reppo", "RecurrentPPO", "MlpLstmPolicy"),
    ("reppo-custom", "RecurrentPPO", CustomRecurrentActorCriticPolicy),
    ("trpo", "TRPO", "MlpPolicy"),
    ("trpo-custom", "TRPO", CustomActorCriticPolicy),
    ("dqn", "DQN", "MlpPolicy"),
    ("dqn-custom", "DQN", CustomDQNPolicy),
    ("qrdqn", "QRDQN", "MlpPolicy"),
    ("qrdqn-custom", "QRDQN", CustomQRDQNPolicy),
    ("a2c", "A2C", "MlpPolicy"),
    ("a2c-custom", "A2C", CustomActorCriticPolicy),
    ("ars", "ARS", "LinearPolicy"),
    ("ars-mlp", "ARS", "MlpPolicy"),
    ("iqn", "IQN", None),  # IQN default-policy branch passes no explicit policy
    ("iqn-custom", "IQN", CustomIQNPolicy),
    ("duel-dqn", "DuelingDQN", None),
    ("duel-dqn-custom", "DuelingDQN", CustomDuelingDQNPolicy),
    ("rainbow-dqn", "RainbowDQN", None),
    ("rainbow-dqn-custom", "RainbowDQN", "__custom_rainbow__"),
    ("munchausen-dqn", "MunchausenDQN", "MlpPolicy"),
    ("munchausen-dqn-custom", "MunchausenDQN", CustomDQNPolicy),
    ("munchausen-duel-dqn-custom", "MunchausenDQN", CustomDuelingDQNPolicy),
]


@pytest.mark.parametrize("model_name, attr, expected_policy", _DISPATCH_CASES)
def test_create_rl_model_selects_right_class_and_policy(monkeypatch, model_name, attr, expected_policy):
    Algo = _patch_algo(monkeypatch, attr)
    cfg = ModelConfig(model_type="rl", model_rl=_rl_config(model_name))
    result = create_rl_model(cfg, env="ENV", device="cpu")
    # Every supported rl branch wraps the algorithm in RLModel.
    assert isinstance(result, RLModel)
    assert isinstance(result.rl_model, Algo)
    if expected_policy == "__custom_rainbow__":
        from src.model.custom.policies import CustomRainbowPolicy

        assert Algo.last["policy"] is CustomRainbowPolicy
    elif expected_policy is None:
        # The branch relies on the algorithm's own default policy — none is passed.
        assert "policy" not in Algo.last
    elif isinstance(expected_policy, str):
        assert Algo.last["policy"] == expected_policy
    else:
        assert Algo.last["policy"] is expected_policy


def test_create_rl_model_maps_core_hyperparameters(monkeypatch):
    Algo = _patch_algo(monkeypatch, "PPO")
    cfg = ModelConfig(model_type="rl", model_rl=_rl_config("ppo"))
    create_rl_model(cfg, env="ENV", device="mps")
    assert Algo.last["env"] == "ENV"
    assert Algo.last["device"] == "mps"
    assert Algo.last["learning_rate"] == 0.0003
    assert Algo.last["batch_size"] == 16
    assert Algo.last["gamma"] == 0.97
    pk = Algo.last["policy_kwargs"]
    # The string lever values are resolved into the concrete torch classes.
    assert pk["optimizer_class"] is torch.optim.Adam
    assert pk["activation_fn"] is torch.nn.ReLU
    assert pk["net_arch"] == [64, 32]
    assert pk["normalize_images"] is False


def test_create_rl_model_resolves_optimizer_and_activation_strings(monkeypatch):
    Algo = _patch_algo(monkeypatch, "DQN")
    cfg = ModelConfig(
        model_type="rl",
        model_rl=_rl_config("dqn", optimizer_class="RMSprop", activation_fn="CELU"),
    )
    create_rl_model(cfg, env="ENV", device="cpu")
    pk = Algo.last["policy_kwargs"]
    assert pk["optimizer_class"] is torch.optim.RMSprop
    assert pk["activation_fn"] is torch.nn.CELU


def test_create_rl_model_custom_net_arch_threaded_for_custom_variants(monkeypatch):
    Algo = _patch_algo(monkeypatch, "DQN")
    cfg = ModelConfig(
        model_type="rl",
        model_rl=_rl_config("dqn-custom", custom_net_arch=["BatchNorm1d", "Linear"]),
    )
    create_rl_model(cfg, env="ENV", device="cpu")
    assert Algo.last["policy_kwargs"]["custom_net_arch"] == ["BatchNorm1d", "Linear"]


def test_create_rl_model_dqn_buffer_and_exploration_threaded(monkeypatch):
    Algo = _patch_algo(monkeypatch, "DQN")
    cfg = ModelConfig(
        model_type="rl",
        model_rl=_rl_config(
            "dqn",
            buffer_size=500,
            tau=0.5,
            exploration_final_eps=0.2,
            exploration_fraction=0.3,
            learning_starts=100,
            train_freq=8,
            gradient_steps=2,
            target_update_interval=250,
            max_grad_norm=0.01,
        ),
    )
    create_rl_model(cfg, env="ENV", device="cpu")
    assert Algo.last["buffer_size"] == 500
    assert Algo.last["tau"] == 0.5
    assert Algo.last["exploration_final_eps"] == 0.2
    assert Algo.last["exploration_fraction"] == 0.3
    assert Algo.last["learning_starts"] == 100
    assert Algo.last["train_freq"] == 8
    assert Algo.last["gradient_steps"] == 2
    assert Algo.last["target_update_interval"] == 250
    assert Algo.last["max_grad_norm"] == 0.01


def test_create_rl_model_lstm_variant_sets_feature_extractor_kwargs(monkeypatch):
    Algo = _patch_algo(monkeypatch, "DuelingDQN")
    cfg = ModelConfig(model_type="rl", model_rl=_rl_config("duel-dqn-custom-lstm"))
    create_rl_model(cfg, env="ENV", device="cpu")
    from src.model.dqn_lstm_policy import LSTMFCE

    pk = Algo.last["policy_kwargs"]
    assert pk["features_extractor_class"] is LSTMFCE
    assert pk["features_extractor_kwargs"]["lstm_hidden_size"] == 2


def test_create_rl_model_lstm3_variant_uses_hidden_size_three(monkeypatch):
    Algo = _patch_algo(monkeypatch, "DuelingDQN")
    cfg = ModelConfig(model_type="rl", model_rl=_rl_config("duel-dqn-custom-lstm3"))
    create_rl_model(cfg, env="ENV", device="cpu")
    assert Algo.last["policy_kwargs"]["features_extractor_kwargs"]["lstm_hidden_size"] == 3


def test_create_rl_model_sets_logger_on_built_model(monkeypatch):
    Algo = _patch_algo(monkeypatch, "PPO")
    cfg = ModelConfig(model_type="rl", model_rl=_rl_config("ppo"))
    result = create_rl_model(cfg, env="ENV", device="cpu")
    assert result.rl_model.set_logger_called is True


def test_create_rl_model_sets_random_seed_when_provided(monkeypatch):
    Algo = _patch_algo(monkeypatch, "PPO")
    cfg = ModelConfig(model_type="rl", model_rl=_rl_config("ppo", seed=123))
    result = create_rl_model(cfg, env="ENV", device="cpu")
    assert result.rl_model.set_seed == 123


def test_create_rl_model_does_not_set_seed_when_none(monkeypatch):
    Algo = _patch_algo(monkeypatch, "PPO")
    cfg = ModelConfig(model_type="rl", model_rl=_rl_config("ppo", seed=None))
    result = create_rl_model(cfg, env="ENV", device="cpu")
    assert result.rl_model.set_seed is None


def test_create_rl_model_loads_checkpoint_when_configured(monkeypatch):
    loaded = {}

    class Algo(_FakeAlgo):
        def load(self, path):
            loaded["path"] = path
            return self

    monkeypatch.setattr(model_factory, "PPO", Algo)
    cfg = ModelConfig(
        model_type="rl",
        model_rl=_rl_config("ppo", checkpoints_folder="ck", checkpoint_to_load="best.zip"),
    )
    create_rl_model(cfg, env="ENV", device="cpu")
    import os

    assert loaded["path"] == os.path.join("ck", "best.zip")


def test_create_rl_model_ensemble_builds_four_members(monkeypatch):
    built = []

    class FakeDueling(_FakeAlgo):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            built.append(kwargs.get("policy"))

    captured_ensemble = {}

    class FakeEnsemble:
        def __init__(self, members):
            captured_ensemble["members"] = members

        def set_logger(self, logger):
            pass

    monkeypatch.setattr(model_factory, "DuelingDQN", FakeDueling)
    monkeypatch.setattr(model_factory, "EnsembleModel", FakeEnsemble)
    cfg = ModelConfig(model_type="rl", model_rl=_rl_config("ensemble"))
    create_rl_model(cfg, env="ENV", device="cpu")
    # Ensemble fans out exactly 4 dueling members, all on the custom policy.
    assert len(captured_ensemble["members"]) == 4
    assert all(p is CustomDuelingDQNPolicy for p in built)


def test_create_rl_model_unsupported_name_raises():
    cfg = ModelConfig(model_type="rl", model_rl=_rl_config("totally-unknown-model"))
    with pytest.raises(ValueError):
        create_rl_model(cfg, env="ENV", device="cpu")


# --------------------------------------------------------------------------------------
# _resolve_pos_weight — BCE positive-class weighting
# --------------------------------------------------------------------------------------

def test_resolve_pos_weight_explicit_value_wins():
    reg = types.SimpleNamespace(pos_weight=3.0)
    env = types.SimpleNamespace(n_positive=10, n_negative=90)
    w = _resolve_pos_weight(reg, env, "cpu")
    assert w is not None
    assert w.item() == pytest.approx(3.0)


def test_resolve_pos_weight_auto_balances_class_ratio():
    # No explicit weight -> n_neg / n_pos.
    reg = types.SimpleNamespace(pos_weight=0.0)
    env = types.SimpleNamespace(n_positive=10, n_negative=90)
    w = _resolve_pos_weight(reg, env, "cpu")
    assert w.item() == pytest.approx(9.0)


def test_resolve_pos_weight_none_when_no_positives():
    reg = types.SimpleNamespace(pos_weight=0.0)
    env = types.SimpleNamespace(n_positive=0, n_negative=90)
    assert _resolve_pos_weight(reg, env, "cpu") is None


def test_resolve_pos_weight_none_when_no_negatives():
    reg = types.SimpleNamespace(pos_weight=0.0)
    env = types.SimpleNamespace(n_positive=10, n_negative=0)
    assert _resolve_pos_weight(reg, env, "cpu") is None


def test_resolve_pos_weight_missing_attrs_default_to_none():
    # Absent pos_weight and absent class counts must not raise -> unweighted (None).
    assert _resolve_pos_weight(types.SimpleNamespace(), types.SimpleNamespace(), "cpu") is None


def test_resolve_pos_weight_negative_configured_falls_through_to_ratio():
    # A non-positive explicit weight is ignored; the ratio branch takes over.
    reg = types.SimpleNamespace(pos_weight=-1.0)
    env = types.SimpleNamespace(n_positive=2, n_negative=8)
    w = _resolve_pos_weight(reg, env, "cpu")
    assert w.item() == pytest.approx(4.0)


# --------------------------------------------------------------------------------------
# create_regression_model — MLP build + loss/optimizer mapping
# --------------------------------------------------------------------------------------

def _reg_config(**overrides):
    base = dict(
        model_name="mlp",
        net_arch=[8, 4],
        custom_net_arch=[],
        loss_fn="mse",
        loss_fn_reduction="mean",
        optimizer_class="Adam",
        learning_rate=0.001,
        seed=None,
    )
    base.update(overrides)
    return ModelRegressionConfig(**base)


def _reg_env(features=6, n_positive=10, n_negative=30):
    return types.SimpleNamespace(
        last_obs=np.zeros(features, dtype=np.float32),
        n_positive=n_positive,
        n_negative=n_negative,
    )


def test_create_regression_model_builds_mlp_regression_model():
    cfg = ModelConfig(model_type="regression", model_regression=_reg_config())
    m = create_regression_model(cfg, _reg_env(), "cpu")
    assert isinstance(m, RegressionModel)
    assert isinstance(m.model, torch.nn.Sequential)


def test_create_regression_model_maps_loss_and_optimizer():
    cfg = ModelConfig(
        model_type="regression",
        model_regression=_reg_config(loss_fn="l1", optimizer_class="AdamW"),
    )
    m = create_regression_model(cfg, _reg_env(), "cpu")
    assert isinstance(m.loss_fn, torch.nn.L1Loss)
    assert isinstance(m.optimizer, torch.optim.AdamW)


def test_create_regression_model_loss_reduction_threaded():
    cfg = ModelConfig(
        model_type="regression",
        model_regression=_reg_config(loss_fn="mse", loss_fn_reduction="none"),
    )
    m = create_regression_model(cfg, _reg_env(), "cpu")
    assert m.loss_fn.reduction == "none"


def test_create_regression_model_optimizer_learning_rate_threaded():
    cfg = ModelConfig(model_type="regression", model_regression=_reg_config(learning_rate=0.05))
    m = create_regression_model(cfg, _reg_env(), "cpu")
    assert m.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_create_regression_model_bcelogits_applies_auto_pos_weight():
    cfg = ModelConfig(
        model_type="regression",
        model_regression=_reg_config(loss_fn="bcelogits"),
    )
    m = create_regression_model(cfg, _reg_env(n_positive=10, n_negative=30), "cpu")
    assert isinstance(m.loss_fn, torch.nn.BCEWithLogitsLoss)
    # n_neg / n_pos = 30 / 10 = 3.
    assert m.loss_fn.pos_weight.item() == pytest.approx(3.0)


def test_create_regression_model_bcelogits_unweighted_when_no_signal():
    cfg = ModelConfig(
        model_type="regression",
        model_regression=_reg_config(loss_fn="bcelogits"),
    )
    m = create_regression_model(cfg, _reg_env(n_positive=0, n_negative=0), "cpu")
    assert isinstance(m.loss_fn, torch.nn.BCEWithLogitsLoss)
    assert m.loss_fn.pos_weight is None


def test_create_regression_model_seed_makes_weights_deterministic():
    cfg = ModelConfig(model_type="regression", model_regression=_reg_config(seed=7))
    m1 = create_regression_model(cfg, _reg_env(), "cpu")
    cfg2 = ModelConfig(model_type="regression", model_regression=_reg_config(seed=7))
    m2 = create_regression_model(cfg2, _reg_env(), "cpu")
    p1 = [p for p in m1.model.parameters() if p.requires_grad][0]
    p2 = [p for p in m2.model.parameters() if p.requires_grad][0]
    assert torch.allclose(p1, p2)


def test_create_regression_model_unsupported_name_raises():
    reg = _reg_config()
    reg.model_name = "not-a-real-regression-model"
    cfg = ModelConfig(model_type="regression", model_regression=reg)
    with pytest.raises(ValueError):
        create_regression_model(cfg, _reg_env(), "cpu")
