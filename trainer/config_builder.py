"""Build BlackSwan's (DataConfig, EnvConfig, ModelConfig) from a flat lever JSON.

Covers the TRADING line (rl + hodl on the `trade_all` env, Sharpe objective).
The dip/trend/regression prediction line uses different envs + an f1-style
objective and belongs in its own manifest. Model configs start from the
repo's tuned `model_rl` instance (reppo-custom / combo_all2 / the 16 reward
multipliers) so a default campaign run matches the best known setup, then
lever values override it; expansion goes through the unchanged
``get_model_combinations`` (which expects OmegaConf-structured nodes).
"""

import copy
import os

from omegaconf import OmegaConf

from src.conf.data_config import (
    DataConfig,
    data_2017_to_2023vs2024_only_price_percent_32_at_1h,
)
from src.conf.env_config import EnvConfig
from src.conf.model_config import ModelConfigSearch, model_rl
from src.model.model_factory import get_model_combinations

_SYMBOL = "BTCUSDT"
_TRAIN_PAIRS = [(y, m) for y in range(2020, 2024) for m in range(1, 13)]
_TEST_PAIRS = [(2024, m) for m in range(1, 5)]


def _daily_files(pairs, symbol=_SYMBOL):
    files = [f"binance/{symbol}-1d-{y}-{m}.json" for (y, m) in pairs]
    return [f for f in files if os.path.exists(f)]


def require_data_present(cfg=None):
    """Fail fast with a clear message when the chosen asset's klines aren't on disk."""
    cfg = cfg or {}
    symbol = str(cfg.get("asset", _SYMBOL))
    if not _daily_files(_TRAIN_PAIRS, symbol) or not _daily_files(_TEST_PAIRS, symbol):
        from trainer.data_inventory import available_assets

        raise SystemExit(
            f"binance/ 1d klines for {symbol} missing — only assets with daily files "
            f"are runnable at 1d. Available at 1d: {available_assets('1d')}."
        )


def _parse_net_arch(value):
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(p) for p in str(value).split(",") if p.strip()]


def build_data_config(cfg):
    asset = str(cfg.get("asset", _SYMBOL))
    timeframe = str(cfg.get("timeframe", "1d"))
    if timeframe == "1h":
        if asset != _SYMBOL:
            raise SystemExit(
                f"{asset} has no 1h dataset on disk — 1h is {_SYMBOL}-only until "
                f"altcoin klines are added (deferred to the data mine)."
            )
        # The repo's tuned 1h instance: 1m source files, downsampled layers.
        return OmegaConf.structured(
            copy.deepcopy(data_2017_to_2023vs2024_only_price_percent_32_at_1h)
        )
    # The env's lookback>1 observation path requires the multi-layer provider;
    # the single-layer daily path therefore runs with lookback 1 (fast,
    # exploratory). The "1h" timeframe is the research-grade path (lookback 32).
    return OmegaConf.structured(
        DataConfig(
            id=f"{asset}-1d-2020to2023vs2024q1",
            train_data_paths=[_daily_files(_TRAIN_PAIRS, asset)],
            test_data_paths=[_daily_files(_TEST_PAIRS, asset)],
            lookback_window_size=1,
            type=str(cfg.get("data_type", "only_price_percent")),
            timestamp="none",
            fidelity_input="1d",
            fidelity_run="1d",
            layers=["1d"],
            fidelity_input_test="1d",
            fidelity_run_test="1d",
            layers_test=["1d"],
        )
    )


def _optional_float(cfg, key, default):
    value = cfg.get(key, default)
    return None if value in (None, "", "null", 0) else float(value)


def build_env_config(cfg):
    return OmegaConf.structured(
        EnvConfig(
            type="trade_all",
            initial_balance=int(cfg.get("initial_balance", 100000)),
            transaction_fee=float(cfg.get("transaction_fee", 0.001)),
            take_profit=_optional_float(cfg, "take_profit", None),
            trailing_take_profit=_optional_float(cfg, "trailing_take_profit", None),
            stop_loss=_optional_float(cfg, "stop_loss", 0.02),
            no_sell_action=bool(cfg.get("no_sell_action", False)),
            observations_contain=[
                "networth_percent_this_trade",
                "in_position",
                "drawdown",
            ],
        )
    )


def is_hodl(cfg):
    return str(cfg.get("model_name", "")).lower() == "hodl" or cfg.get("model_type") == "hodl"


def build_model_config(cfg):
    """Return one concrete ModelConfig for the lever values in ``cfg``."""
    if is_hodl(cfg):
        hodl = OmegaConf.structured(ModelConfigSearch(model_type="hodl"))
        config = get_model_combinations(hodl)[0]
        config.iterations_to_pick_best = 1
        return config

    search = copy.deepcopy(model_rl)
    rl = search.model_rl
    model_name = str(cfg.get("model_name", "reppo-custom"))
    rl.model_name = [model_name]
    if not model_name.endswith("-custom"):
        # The tuned custom_net_arch tokens only apply to the *-custom models.
        rl.custom_net_arch = [[]]
    rl.reward_model = [str(cfg.get("reward_model", "combo_all2"))]
    rl.learning_rate = [float(cfg.get("learning_rate", 0.0001))]
    rl.gamma = [float(cfg.get("gamma", 0.99))]
    rl.batch_size = [int(cfg.get("batch_size", 512))]
    rl.buffer_size = [int(cfg.get("buffer_size", 100000))]
    rl.learning_starts = [int(cfg.get("learning_starts", 1000))]
    rl.episodes = [int(cfg.get("episodes", 1))]
    rl.seed = int(cfg["seed"]) if cfg.get("seed") is not None else None
    if cfg.get("checkpoint_to_load"):
        rl.checkpoint_to_load = str(cfg["checkpoint_to_load"])
    if "net_arch" in cfg:
        rl.net_arch = [_parse_net_arch(cfg["net_arch"])]
    if "optimizer_class" in cfg:
        rl.optimizer_class = [str(cfg["optimizer_class"])]
    if "activation_fn" in cfg:
        rl.activation_fn = [str(cfg["activation_fn"])]
    config = get_model_combinations(OmegaConf.structured(search))[0]
    config.iterations_to_pick_best = 1
    return config
