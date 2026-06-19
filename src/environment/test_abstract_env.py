"""Direct tests for ``AbstractEnv``'s concrete helpers (``get_id``, ``__init__`` defaults, ``setup``,
``get_price``, ``get_timesteps``).

We define a minimal concrete subclass implementing the abstract methods as no-ops so the class can be
instantiated, and stub the data provider. The abstract method bodies themselves are trivial ``pass``
stubs and are not behaviour-bearing (covered only incidentally); they are noted in the agent output.
"""

import types

import pandas as pd
from gymnasium import spaces

from src.conf.env_config import EnvConfig
from src.environment.abstract_env import AbstractEnv


class _ConcreteEnv(AbstractEnv):
    """Smallest instantiable AbstractEnv: every abstract method is a trivial stub."""

    def create_action_space(self):
        return spaces.Discrete(2)

    def create_observation_space(self):
        return spaces.Box(low=-1, high=1, shape=(1,))

    def take_action(self, action):
        return False

    def calculate_cumulative_return(self):
        return 0.0

    def calculate_sharpe_ratio(self, risk_free_rate):
        return 0.0

    def plot_portfolio_value(self):
        return None

    def log_metrics_over_time(self, prefix, log_remote):
        return None

    def get_run_state(self):
        return []

    def get_raw_df_for_plotting(self):
        return pd.DataFrame()

    # Avoid the gymnasium.Env.reset chain (setup() calls reset()); keep it inert for the setup test.
    def reset(self, seed=None, options=None):
        return None, {}


def _provider(*, timesteps=42, price=123.0):
    return types.SimpleNamespace(
        get_timesteps=lambda: timesteps,
        get_price=lambda step: price + step,
    )


def _env(config, provider=None, device="cpu"):
    return _ConcreteEnv(config, provider or _provider(), device)


def test_init_sets_core_attributes_and_defaults():
    cfg = EnvConfig(type="trade_all")
    provider = _provider()
    e = _env(cfg, provider, device="cuda:1")
    assert e.env_config is cfg
    assert e.data_provider is provider
    assert e.device == "cuda:1"
    # Documented defaults set in __init__.
    assert e.reward_model == "percent_profit"
    assert e.current_step == 0
    # id is computed at construction time.
    assert e.id == e.get_id()


def test_get_timesteps_delegates_to_provider():
    e = _env(EnvConfig(type="trade_all"), _provider(timesteps=99))
    assert e.get_timesteps() == 99


def test_get_price_delegates_to_provider_with_step():
    e = _env(EnvConfig(type="trade_all"), _provider(price=200.0))
    # provider stub returns price + step.
    assert e.get_price(0) == 200.0
    assert e.get_price(5) == 205.0


def test_setup_stores_reward_model_and_multipliers():
    e = _env(EnvConfig(type="trade_all"))
    multipliers = {"combo_sell": 1.0}
    e.setup("combo_all", multipliers)
    assert e.reward_model == "combo_all"
    assert e.reward_multipliers is multipliers


def test_get_id_includes_type_and_none_defaults_as_zero():
    # take_profit/stop_loss/trailing default to None -> get_id substitutes 0.
    cfg = EnvConfig(type="swap", amount=1, transaction_fee=0.001, batch_size=32)
    e = _env(cfg)
    gid = e.get_id()
    assert gid.startswith("swap_1_")
    # transaction_fee 0.001 * 100 = 0.1 -> '~' replaces '.', so '-0~1%' appears.
    assert "-0~1%" in gid
    assert "TP0" in gid and "TR0" in gid and "SL0" in gid
    assert "B32" in gid


def test_get_id_uses_explicit_tpsl_values():
    cfg = EnvConfig(
        type="trade_all",
        take_profit=0.02,
        trailing_take_profit=0.01,
        stop_loss=0.03,
    )
    gid = _env(cfg).get_id()
    # 0.02 -> '0~02' etc. (dots become '~').
    assert "TP0~02" in gid
    assert "TR0~01" in gid
    assert "SL0~03" in gid


def test_get_id_replaces_dots_with_tilde():
    cfg = EnvConfig(type="trade_all", transaction_fee=0.005)
    gid = _env(cfg).get_id()
    assert "." not in gid  # every dot must be rewritten to '~'


def test_get_id_counts_observations_contain_length():
    cfg = EnvConfig(type="trade_all", observations_contain=["a", "b", "c"])
    gid = _env(cfg).get_id()
    # Format: {type}_{amount}_{len(observations_contain)}_...
    assert "_3_" in gid


def test_get_id_zero_transaction_fee_when_none():
    # transaction_fee None -> the env_config field is typed float, but get_id guards for None -> 0.
    cfg = EnvConfig(type="trade_all")
    cfg.transaction_fee = None
    gid = _env(cfg).get_id()
    # 0% (no decimal) for the fee segment.
    assert "_-0%_" in gid


def test_abstract_methods_are_flagged_abstract():
    # The methods carry the @abstractmethod marker even though it is not enforced (see below).
    assert AbstractEnv.create_action_space.__isabstractmethod__ is True
    assert AbstractEnv.take_action.__isabstractmethod__ is True
    assert AbstractEnv.calculate_sharpe_ratio.__isabstractmethod__ is True


def test_abstract_env_is_instantiable_despite_abstractmethods():
    # FOOTGUN: gymnasium.Env does NOT use ABCMeta, so @abstractmethod is decorative only — AbstractEnv
    # can be instantiated directly and its abstract methods return None instead of raising. Pin the
    # actual (non-enforcing) behaviour so a regression to real ABC enforcement is noticed.
    e = AbstractEnv(EnvConfig(type="trade_all"), _provider(), "cpu")
    assert e.create_action_space() is None
    assert e.take_action(1) is None
    assert e.get_run_state() is None
