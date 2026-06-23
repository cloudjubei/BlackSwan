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

from src.conf.data_config import DataConfig
from src.conf.env_config import EnvConfig
from src.conf.model_config import ModelConfig, ModelConfigSearch, ModelSupervisedConfig, model_rl
from src.model.model_factory import get_model_combinations
from trainer.fidelity import resolve_fidelity
from trainer.walk_forward import resolve_walk_forward_window

_SYMBOL = "BTCUSDT"
# The default walk-forward window ("2024") resolves to exactly these pairs; the dip line
# (trainer/dip.py) reads them directly, so they stay as the canonical single-split default.
_TRAIN_PAIRS = [(y, m) for y in range(2020, 2024) for m in range(1, 13)]
_TEST_PAIRS = [(2024, m) for m in range(1, 13)]


def _daily_files(pairs, symbol=_SYMBOL):
    files = [f"binance/{symbol}-1d-{y}-{m}.json" for (y, m) in pairs]
    return [f for f in files if os.path.exists(f)]


def require_data_present(cfg=None):
    """Fail fast with a clear message when the chosen asset's klines aren't on disk."""
    cfg = cfg or {}
    symbol = str(cfg.get("asset", _SYMBOL))
    train_pairs, test_pairs, _ = resolve_walk_forward_window(cfg)
    if not _daily_files(train_pairs, symbol) or not _daily_files(test_pairs, symbol):
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
    train_pairs, test_pairs, window = resolve_walk_forward_window(cfg)
    wf = window["walk_forward_window"]
    fset_id, fspec = resolve_fidelity(cfg)
    layers = list(fspec["layers"])
    lookback = int(fspec["lookback"])
    # Intraday sets step on 1h bars (with higher layers resampled by the multi-layer provider) and
    # need the derived-cache path; the single daily set runs the fast lookback-1 path off raw 1d files.
    if fspec["fidelity_run"] != "1d":
        if asset != _SYMBOL:
            raise SystemExit(
                f"{asset} has no intraday dataset on disk — intraday is {_SYMBOL}-only until "
                f"altcoin klines are added (deferred to the data mine)."
            )
        from trainer.derive_cache import ensure_derived

        return OmegaConf.structured(
            DataConfig(
                id=f"{asset}-{fset_id}-wf{wf}",
                train_data_paths=[ensure_derived(asset, train_pairs, "1h")],
                test_data_paths=[ensure_derived(asset, test_pairs, "1h")],
                lookback_window_size=int(cfg.get("lookback_window") or lookback),
                type=str(cfg.get("data_type", "only_price_percent")),
                use_indicators=bool(cfg.get("use_indicators", False)),
                timestamp="day_of_week",
                obs_squash=str(cfg.get("obs_squash", "none")),
                fidelity_input="1h",
                fidelity_run="1h",
                layers=layers,
                fidelity_input_test="1h",
                fidelity_run_test="1h",
                layers_test=layers,
            )
        )
    return OmegaConf.structured(
        DataConfig(
            id=f"{asset}-{fset_id}-wf{wf}",
            train_data_paths=[_daily_files(train_pairs, asset)],
            test_data_paths=[_daily_files(test_pairs, asset)],
            lookback_window_size=lookback,
            type=str(cfg.get("data_type", "only_price_percent")),
            use_indicators=bool(cfg.get("use_indicators", False)),
            timestamp="none",
            obs_squash=str(cfg.get("obs_squash", "none")),
            fidelity_input="1d",
            fidelity_run="1d",
            layers=layers,
            fidelity_input_test="1d",
            fidelity_run_test="1d",
            layers_test=layers,
        )
    )


def _optional_float(cfg, key, default):
    value = cfg.get(key, default)
    return None if value in (None, "", "null", 0) else float(value)


def _resolve_position_mode(cfg):
    """position_mode is the authoritative 3-way lever; fall back to the legacy `allow_shorting` bool when
    it is absent (old configs/runs). Returns (position_mode, allow_shorting) kept consistent."""
    mode = str(cfg.get("position_mode", "")).strip().lower()
    if mode not in ("long_only", "short_only", "both"):
        mode = "both" if bool(cfg.get("allow_shorting", False)) else "long_only"
    return mode, mode in ("short_only", "both")


def build_env_config(cfg):
    position_mode, allow_shorting = _resolve_position_mode(cfg)
    return OmegaConf.structured(
        EnvConfig(
            type="trade_all",
            initial_balance=int(cfg.get("initial_balance", 100000)),
            transaction_fee=float(cfg.get("transaction_fee", 0.001)),
            take_profit=_optional_float(cfg, "take_profit", None),
            trailing_take_profit=_optional_float(cfg, "trailing_take_profit", None),
            stop_loss=_optional_float(cfg, "stop_loss", 0.02),
            no_sell_action=bool(cfg.get("no_sell_action", False)),
            position_sizing=str(cfg.get("position_sizing", "fixed")),
            vol_target=float(cfg.get("vol_target", 0.02)),
            vol_target_min=float(cfg.get("vol_target_min", 0.1)),
            vol_window=int(cfg.get("vol_window", 10)),
            allow_shorting=allow_shorting,
            position_mode=position_mode,
            max_short_size=float(cfg.get("max_short_size", 1.0)),
            observations_contain=[
                "networth_percent_this_trade",
                "in_position",
                "drawdown",
            ],
        )
    )


def is_hodl(cfg):
    return str(cfg.get("model_name", "")).lower() == "hodl" or cfg.get("model_type") == "hodl"


def is_supervised(cfg):
    return str(cfg.get("model_name", "")).lower().startswith("supervised") or cfg.get("model_type") == "supervised"


def build_model_config(cfg):
    """Return one concrete ModelConfig for the lever values in ``cfg``."""
    if is_hodl(cfg):
        hodl = OmegaConf.structured(ModelConfigSearch(model_type="hodl"))
        config = get_model_combinations(hodl)[0]
        config.iterations_to_pick_best = 1
        return config

    if is_supervised(cfg):
        supervised = ModelSupervisedConfig(
            model_name=str(cfg.get("model_name", "supervised-logreg")),
            forward_horizon=int(cfg.get("forward_horizon", 1)),
            prob_threshold=float(cfg.get("prob_threshold", 0.5)),
            seed=int(cfg["seed"]) if cfg.get("seed") is not None else None,
        )
        config = ModelConfig(model_type="supervised", model_supervised=supervised)
        config.iterations_to_pick_best = 1
        return config

    search = copy.deepcopy(model_rl)
    rl = search.model_rl
    model_name = str(cfg.get("model_name", "reppo-custom"))
    rl.model_name = [model_name]
    if not model_name.endswith("-custom"):
        # The tuned custom_net_arch tokens only apply to the *-custom models.
        rl.custom_net_arch = [[]]
    rl.reward_model = [str(cfg.get("reward_model", "combo_unified"))]
    rl.reward_multiplier_combo_noaction = [float(cfg.get("combo_noaction", 0))]
    # combo_unified defaults the two penalties OFF (0) so a bare combo_unified run ≡ the old combo_all.
    rl.reward_multiplier_combo_fee_penalty = [float(cfg.get("combo_fee_penalty", 0))]
    rl.reward_multiplier_combo_noop_penalty = [float(cfg.get("combo_noop_penalty", 0))]
    # combo_unified exposes the remaining combo weights as levers. Only override the baked model_rl value
    # when the run actually carries the key, so the existing reward models are unchanged: a migrated
    # combo_all run sets combo_wrongaction=0 (combo_unified then adds nothing), while a migrated combo_all2
    # run omits it (inheriting the baked value, which combo_unified adds — reproducing combo_all2).
    for _k in ("combo_sell", "combo_buy", "combo_positionprofitpercentage", "combo_wrongaction", "combo_direct"):
        if _k in cfg:
            setattr(rl, "reward_multiplier_" + _k, [float(cfg[_k])])
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
    if "exploration_fraction" in cfg:
        rl.exploration_fraction = [float(cfg["exploration_fraction"])]
    if "exploration_final_eps" in cfg:
        rl.exploration_final_eps = [float(cfg["exploration_final_eps"])]
    config = get_model_combinations(OmegaConf.structured(search))[0]
    config.iterations_to_pick_best = 1
    return config
