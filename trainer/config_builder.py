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
from src.conf.model_config import ModelConfig, ModelConfigSearch, ModelSupervisedConfig, ModelMomentumConfig, ModelMaCrossoverConfig, ModelBreakoutConfig, ModelTimeConfig, ModelDayConfig, ModelTechnicalConfig, model_rl
from src.model.model_factory import get_model_combinations
from src.model.rl_model import is_recurrent_model_name
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


def _minute_files(pairs, symbol=_SYMBOL):
    # The canonical 1m source klines (NOT derived — 1m IS the source of truth; coarser layers resample
    # from it at runtime in the provider). Used by the 1m-base path.
    files = [f"binance/{symbol}-1m-{y}-{m}.json" for (y, m) in pairs]
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
    # The technical baseline reads curated indicator columns (rsi10, ...) by name, so enable them even when
    # the user left use_indicators off — otherwise the column lookup would fail at run time.
    use_indicators = bool(cfg.get("use_indicators", False)) or is_technical(cfg)
    train_pairs, test_pairs, window = resolve_walk_forward_window(cfg)
    wf = window["walk_forward_window"]
    fset_id, fspec = resolve_fidelity(cfg)
    layers = list(fspec["layers"])
    lookback = int(fspec["lookback"])
    fidelity_run = fspec["fidelity_run"]
    # The 1m BASE path serves any run whose finest involved granularity is 1m — a minute step, OR a
    # coarser step (1h/1d) over the 1m base where the provider's divider_run advances the right number of
    # 1m bars per decision (observe 1m micro-structure, decide coarsely). It reads the RAW 1m source
    # directly (no derive: 1m is the source of truth; coarser layers resample at runtime).
    if fspec["fidelity_input"] == "1m":
        if asset != _SYMBOL:
            raise SystemExit(
                f"{asset} has no 1-minute dataset on disk — minute data is {_SYMBOL}-only until "
                f"altcoin klines are added (deferred to the data mine)."
            )
        train_files = _minute_files(train_pairs, asset)
        test_files = _minute_files(test_pairs, asset)
        if not train_files or not test_files:
            raise SystemExit(
                f"binance/ 1m klines for {asset} missing for window {wf} — a 1m run needs the raw "
                f"minute source on disk."
            )
        return OmegaConf.structured(
            DataConfig(
                id=f"{asset}-{fset_id}-wf{wf}",
                train_data_paths=[train_files],
                test_data_paths=[test_files],
                lookback_window_size=int(cfg.get("lookback_window") or lookback),
                type=str(cfg.get("data_type", "only_price_percent")),
                use_indicators=use_indicators,
                timestamp="day_of_week",
                obs_squash=str(cfg.get("obs_squash", "none")),
                fidelity_input="1m",
                fidelity_run=fidelity_run,
                layers=layers,
                fidelity_input_test="1m",
                fidelity_run_test=fidelity_run,
                layers_test=layers,
            )
        )
    # The 1h BASE path serves any step that observes a 1h layer — an hourly step AND a daily step over the
    # 1h base (stepped day by day, the provider's divider_run handles the cadence). It needs the derived
    # cache. The 1d base path runs off raw 1d files (single 1d, or 1d+1w resampled from 1d).
    if fspec["fidelity_input"] == "1h":
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
                use_indicators=use_indicators,
                timestamp="day_of_week",
                obs_squash=str(cfg.get("obs_squash", "none")),
                fidelity_input="1h",
                fidelity_run=fidelity_run,
                layers=layers,
                fidelity_input_test="1h",
                fidelity_run_test=fidelity_run,
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
            use_indicators=use_indicators,
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
            position_sizing=str(cfg.get("position_sizing", "fixed")),
            vol_target=float(cfg.get("vol_target", 0.02)),
            vol_target_min=float(cfg.get("vol_target_min", 0.1)),
            vol_window=int(cfg.get("vol_window", 10)),
            allow_shorting=bool(cfg.get("allow_shorting", False)),
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


def is_momentum(cfg):
    return str(cfg.get("model_name", "")).lower() == "momentum" or cfg.get("model_type") == "momentum"


def is_ma_crossover(cfg):
    return str(cfg.get("model_name", "")).lower() == "ma_crossover" or cfg.get("model_type") == "ma_crossover"


def is_breakout(cfg):
    return str(cfg.get("model_name", "")).lower() == "breakout" or cfg.get("model_type") == "breakout"


def is_time(cfg):
    return str(cfg.get("model_name", "")).lower() == "time" or cfg.get("model_type") == "time"


def is_weekday(cfg):
    return str(cfg.get("model_name", "")).lower() == "weekday" or cfg.get("model_type") == "weekday"


def is_technical(cfg):
    return str(cfg.get("model_name", "")).lower() == "technical" or cfg.get("model_type") == "technical"


# Master switch for persisting model CHECKPOINTS (the large .zip / .pt weight files). DISABLED project-wide to
# save disk — the save code stays in every model, gated on `config.save_checkpoint`. Re-enable for the whole
# project by flipping this to True, or per run by passing cfg["save_checkpoint"]=True. (Note: with checkpoints
# off, `--evaluate` / checkpoint-replay have nothing to load — the xAI decision trace is unaffected.)
SAVE_CHECKPOINTS = False


def build_model_config(cfg):
    """One concrete ModelConfig for ``cfg``, with the project-wide checkpoint switch applied to every path."""
    config = _build_model_config(cfg)
    config.save_checkpoint = bool(cfg.get("save_checkpoint", SAVE_CHECKPOINTS))
    return config


def _build_model_config(cfg):
    """Return one concrete ModelConfig for the lever values in ``cfg``."""
    if is_hodl(cfg):
        hodl = OmegaConf.structured(ModelConfigSearch(model_type="hodl"))
        config = get_model_combinations(hodl)[0]
        config.iterations_to_pick_best = 1
        return config

    if is_momentum(cfg):
        momentum = ModelMomentumConfig(lookback_periods=int(cfg.get("momentum_lookback", 30)))
        config = ModelConfig(model_type="momentum", model_momentum=momentum)
        config.iterations_to_pick_best = 1
        return config

    if is_ma_crossover(cfg):
        # Variable-length MA crossover (BLL 1992 / Grobys 2020): long when the short MA is above the long
        # MA by more than the band, flat when below, hold inside. Windows are in BARS (use a daily timeframe
        # for the papers' daily 1/50, 1/150, 1/200, 1/20 rules).
        ma = ModelMaCrossoverConfig(
            short_window=int(cfg.get("ma_short_window", 1)),
            long_window=int(cfg.get("ma_long_window", 50)),
            band=float(cfg.get("ma_band", 0.0)),
        )
        config = ModelConfig(model_type="ma_crossover", model_ma_crossover=ma)
        config.iterations_to_pick_best = 1
        return config

    if is_breakout(cfg):
        # Trading-range breakout (BLL 1992): long when price breaks above the prior `window`-bar high by
        # more than the band, flat when it breaks below the low. `window` is in BARS (daily 50/150/200).
        breakout = ModelBreakoutConfig(
            window=int(cfg.get("breakout_window", 50)),
            band=float(cfg.get("breakout_band", 0.0)),
        )
        config = ModelConfig(model_type="breakout", model_breakout=breakout)
        config.iterations_to_pick_best = 1
        return config

    if is_time(cfg):
        # Deterministic time-of-day baseline: long from UTC hour `time_buy` to hour `time_sell`. Needs
        # intraday (1h) data so the hour varies.
        time_cfg = ModelTimeConfig(
            time_buy=int(cfg.get("time_buy", 14)),
            time_sell=int(cfg.get("time_sell", 21)),
        )
        config = ModelConfig(model_type="time", model_time=time_cfg)
        config.iterations_to_pick_best = 1
        return config

    if is_weekday(cfg):
        # Deterministic day-of-week baseline: long from weekday `day_buy` (0=Mon..6=Sun) to `day_sell`.
        # Runs at a daily step (timeframe=1d) so the weekday is unambiguous.
        day_cfg = ModelDayConfig(
            day_buy=int(cfg.get("day_buy", 0)),
            day_sell=int(cfg.get("day_sell", 4)),
        )
        config = ModelConfig(model_type="weekday", model_day=day_cfg)
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

    if is_technical(cfg):
        # Deterministic technical-indicator baseline: buy/sell when a chosen [-1,1]-normalised indicator
        # crosses its threshold. Defaults to classic RSI mean-reversion (oversold buy / overbought sell);
        # the direction flags in ModelTechnicalConfig (buy_is_down_check / sell_is_up_check) stay at their
        # mean-reversion defaults. Needs the curated indicator columns (build_data_config forces them on).
        technical = ModelTechnicalConfig(
            buy_indicator=str(cfg.get("technical_buy_indicator", "rsi10")),
            buy_amount_threshold=float(cfg.get("technical_buy_threshold", -0.4)),
            sell_indicator=str(cfg.get("technical_sell_indicator", "rsi10")),
            sell_amount_threshold=float(cfg.get("technical_sell_threshold", 0.4)),
            # Direction flags pick the rule FAMILY: default (down/up) = mean-reversion (buy oversold, sell
            # overbought); both off = trend-following (buy above / sell below, e.g. trendSlope10 crossing 0).
            buy_is_down_check=bool(cfg.get("technical_buy_is_down_check", True)),
            sell_is_up_check=bool(cfg.get("technical_sell_is_up_check", True)),
        )
        config = ModelConfig(model_type="technical", model_technical=technical)
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
    if "lstm_hidden_size" in cfg:
        rl.lstm_hidden_size = [int(cfg["lstm_hidden_size"])]
    if "shared_lstm" in cfg:
        rl.shared_lstm = [bool(cfg["shared_lstm"])]
    if "enable_critic_lstm" in cfg:
        rl.enable_critic_lstm = [bool(cfg["enable_critic_lstm"])]
    if "ssm_state_dim" in cfg:
        rl.ssm_state_dim = [int(cfg["ssm_state_dim"])]
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
    if is_recurrent_model_name(model_name) and config.model_rl.shared_lstm and config.model_rl.enable_critic_lstm:
        raise SystemExit(
            "shared_lstm=True requires enable_critic_lstm=False — SB3's RecurrentActorCriticPolicy "
            "shares ONE LSTM across actor+critic only when the separate critic LSTM is disabled."
        )
    return config
