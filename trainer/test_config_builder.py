import pytest

from trainer import config_builder


def _echo_daily(monkeypatch):
    monkeypatch.setattr(
        config_builder,
        "_daily_files",
        lambda pairs, symbol=config_builder._SYMBOL: [f"{y}-{m}" for (y, m) in pairs],
    )


def test_build_data_config_1d_uses_the_selected_window_pairs(monkeypatch):
    _echo_daily(monkeypatch)
    cfg = config_builder.build_data_config({"timeframe": "1d", "walk_forward_window": "2022"})
    train = list(cfg.train_data_paths[0])
    test = list(cfg.test_data_paths[0])
    assert train[0] == "2020-1"
    assert train[-1] == "2021-12"
    assert test == [f"2022-{m}" for m in range(1, 13)]
    assert "2022" in cfg.id


def test_build_data_config_1d_default_window_is_the_legacy_split(monkeypatch):
    _echo_daily(monkeypatch)
    cfg = config_builder.build_data_config({"timeframe": "1d"})
    train = list(cfg.train_data_paths[0])
    test = list(cfg.test_data_paths[0])
    assert train[0] == "2020-1"
    assert train[-1] == "2023-12"
    assert test == [f"2024-{m}" for m in range(1, 13)]


def test_build_data_config_1h_uses_window_and_stays_btc_only(monkeypatch):
    import trainer.derive_cache as dc

    monkeypatch.setattr(
        dc,
        "ensure_derived",
        lambda symbol, pairs, fidelity, cache_dir=None: [f"{fidelity}-{y}-{m}" for (y, m) in pairs],
    )
    cfg = config_builder.build_data_config({"timeframe": "1h", "walk_forward_window": "2023"})
    train = list(cfg.train_data_paths[0])
    test = list(cfg.test_data_paths[0])
    assert train[0] == "1h-2020-1"
    assert train[-1] == "1h-2022-12"
    assert test == [f"1h-2023-{m}" for m in range(1, 13)]
    assert "2023" in cfg.id


def test_build_data_config_fidelity_set_stacks_layers(monkeypatch):
    import trainer.derive_cache as dc

    monkeypatch.setattr(
        dc,
        "ensure_derived",
        lambda symbol, pairs, fidelity, cache_dir=None: [f"{fidelity}-{y}-{m}" for (y, m) in pairs],
    )
    cfg = config_builder.build_data_config(
        {"timeframe": "1h", "fidelity_set": "1h+1d+1w", "walk_forward_window": "2024"}
    )
    assert list(cfg.layers) == ["1h", "1d", "1w"]
    assert list(cfg.layers_test) == ["1h", "1d", "1w"]
    assert cfg.lookback_window_size == 32
    assert "1h+1d+1w" in cfg.id


def test_build_data_config_coarser_only_stack_at_hourly_step(monkeypatch):
    import trainer.derive_cache as dc

    monkeypatch.setattr(
        dc, "ensure_derived", lambda symbol, pairs, fidelity, cache_dir=None: ["f"]
    )
    # An hourly-stepping agent fed only coarser layers (1d, 1w) every hour.
    cfg = config_builder.build_data_config({"timeframe": "1h", "fidelity_set": "1d+1w"})
    assert list(cfg.layers) == ["1d", "1w"]
    assert cfg.fidelity_run == "1h"
    assert cfg.lookback_window_size == 32


def test_build_data_config_daily_step_with_finer_layers_uses_1h_base(monkeypatch):
    # A daily step observing finer (1h) layers loads the 1h derived base and steps it day by day
    # (fidelity_run='1d' over fidelity_input='1h') instead of failing fast.
    import trainer.derive_cache as dc

    monkeypatch.setattr(dc, "ensure_derived", lambda symbol, pairs, fidelity, cache_dir=None: ["f"])
    cfg = config_builder.build_data_config({"timeframe": "1d", "fidelity_set": "1h+1d"})
    assert list(cfg.layers) == ["1h", "1d"]
    assert cfg.fidelity_input == "1h"
    assert cfg.fidelity_run == "1d"
    assert cfg.fidelity_input_test == "1h"
    assert cfg.fidelity_run_test == "1d"
    assert cfg.lookback_window_size == 32


def test_build_data_config_1m_uses_raw_minute_source(monkeypatch):
    # B0: a minute-base run reads the RAW 1m files directly (no derive cache); coarser layers resample
    # from them at runtime.
    monkeypatch.setattr(
        config_builder,
        "_minute_files",
        lambda pairs, symbol=config_builder._SYMBOL: [f"1m-{y}-{m}" for (y, m) in pairs],
    )
    cfg = config_builder.build_data_config({"timeframe": "1m", "walk_forward_window": "2024"})
    assert cfg.fidelity_input == "1m"
    assert cfg.fidelity_run == "1m"
    assert list(cfg.layers) == ["1m", "1h"]
    assert list(cfg.train_data_paths[0])[0] == "1m-2020-1"
    assert "1m+1h" in cfg.id
    assert cfg.lookback_window_size == 32


def test_build_data_config_1m_base_with_hourly_decision_cadence(monkeypatch):
    # B2: observe 1m micro-structure, DECIDE hourly -> 1m base (fidelity_input), 1h step (fidelity_run).
    monkeypatch.setattr(
        config_builder, "_minute_files", lambda pairs, symbol=config_builder._SYMBOL: ["f"]
    )
    cfg = config_builder.build_data_config({"timeframe": "1h", "fidelity_set": "1m+1h"})
    assert cfg.fidelity_input == "1m"
    assert cfg.fidelity_run == "1h"
    assert list(cfg.layers) == ["1m", "1h"]


def test_build_data_config_1m_non_btc_fails_fast():
    with pytest.raises(SystemExit):
        config_builder.build_data_config({"asset": "ETHUSDT", "timeframe": "1m"})


def test_build_data_config_1m_missing_source_fails_fast(monkeypatch):
    monkeypatch.setattr(config_builder, "_minute_files", lambda pairs, symbol=config_builder._SYMBOL: [])
    with pytest.raises(SystemExit, match="1m klines"):
        config_builder.build_data_config({"timeframe": "1m"})


def test_build_data_config_default_1h_is_the_1h_plus_1d_stack(monkeypatch):
    import trainer.derive_cache as dc

    monkeypatch.setattr(
        dc, "ensure_derived", lambda symbol, pairs, fidelity, cache_dir=None: ["f"]
    )
    cfg = config_builder.build_data_config({"timeframe": "1h"})
    assert list(cfg.layers) == ["1h", "1d"]
    assert cfg.lookback_window_size == 32


def test_build_data_config_lookback_window_override(monkeypatch):
    import trainer.derive_cache as dc

    monkeypatch.setattr(
        dc, "ensure_derived", lambda symbol, pairs, fidelity, cache_dir=None: ["f"]
    )
    cfg = config_builder.build_data_config({"timeframe": "1h", "lookback_window": 64})
    assert cfg.lookback_window_size == 64


def test_is_supervised_detects_supervised_model_names():
    assert config_builder.is_supervised({"model_name": "supervised-logreg"})
    assert config_builder.is_supervised({"model_name": "supervised-gbm"})
    assert config_builder.is_supervised({"model_type": "supervised"})
    assert not config_builder.is_supervised({"model_name": "reppo-custom"})
    assert not config_builder.is_supervised({"model_name": "hodl"})


def test_build_model_config_supervised_maps_levers():
    config = config_builder.build_model_config(
        {"model_name": "supervised-gbm", "forward_horizon": 3, "prob_threshold": 0.6, "seed": 7}
    )
    assert config.model_type == "supervised"
    assert config.model_supervised.model_name == "supervised-gbm"
    assert config.model_supervised.forward_horizon == 3
    assert config.model_supervised.prob_threshold == 0.6
    assert config.model_supervised.seed == 7


def test_checkpoints_disabled_project_wide_by_default():
    # The project master switch (SAVE_CHECKPOINTS) is off, so every model type builds with save_checkpoint=False.
    assert config_builder.SAVE_CHECKPOINTS is False
    for cfg in (
        {"model_name": "ppo-custom", "seed": 0},
        {"model_name": "supervised-gbm", "seed": 0},
        {"model_name": "hodl"},
        {"model_name": "momentum"},
    ):
        assert config_builder.build_model_config(cfg).save_checkpoint is False


def test_save_checkpoint_can_be_re_enabled_per_run():
    # The flag stays in code — a run may opt back in without touching the master switch.
    config = config_builder.build_model_config(
        {"model_name": "ppo-custom", "seed": 0, "save_checkpoint": True}
    )
    assert config.save_checkpoint is True


def test_is_momentum_detects_momentum_model():
    assert config_builder.is_momentum({"model_name": "momentum"})
    assert config_builder.is_momentum({"model_type": "momentum"})
    assert not config_builder.is_momentum({"model_name": "hodl"})
    assert not config_builder.is_momentum({"model_name": "supervised-gbm"})


def test_build_model_config_momentum_maps_lookback():
    config = config_builder.build_model_config({"model_name": "momentum", "momentum_lookback": 252})
    assert config.model_type == "momentum"
    assert config.model_momentum.lookback_periods == 252


def test_build_model_config_momentum_defaults_lookback():
    config = config_builder.build_model_config({"model_name": "momentum"})
    assert config.model_type == "momentum"
    assert config.model_momentum.lookback_periods == 30


def test_is_ma_crossover_detects_ma_crossover_model():
    assert config_builder.is_ma_crossover({"model_name": "ma_crossover"})
    assert config_builder.is_ma_crossover({"model_type": "ma_crossover"})
    assert not config_builder.is_ma_crossover({"model_name": "momentum"})


def test_build_model_config_ma_crossover_maps_windows_and_band():
    config = config_builder.build_model_config(
        {"model_name": "ma_crossover", "ma_short_window": 1, "ma_long_window": 200, "ma_band": 0.01}
    )
    assert config.model_type == "ma_crossover"
    assert config.model_ma_crossover.short_window == 1
    assert config.model_ma_crossover.long_window == 200
    assert config.model_ma_crossover.band == 0.01


def test_build_model_config_ma_crossover_defaults():
    config = config_builder.build_model_config({"model_name": "ma_crossover"})
    assert config.model_type == "ma_crossover"
    assert config.model_ma_crossover.short_window == 1
    assert config.model_ma_crossover.long_window == 50
    assert config.model_ma_crossover.band == 0.0


def test_is_breakout_detects_breakout_model():
    assert config_builder.is_breakout({"model_name": "breakout"})
    assert config_builder.is_breakout({"model_type": "breakout"})
    assert not config_builder.is_breakout({"model_name": "ma_crossover"})


def test_build_model_config_breakout_maps_window_and_band():
    config = config_builder.build_model_config(
        {"model_name": "breakout", "breakout_window": 150, "breakout_band": 0.02}
    )
    assert config.model_type == "breakout"
    assert config.model_breakout.window == 150
    assert config.model_breakout.band == 0.02


def test_build_model_config_breakout_defaults():
    config = config_builder.build_model_config({"model_name": "breakout"})
    assert config.model_type == "breakout"
    assert config.model_breakout.window == 50
    assert config.model_breakout.band == 0.0


def test_is_technical_detects_technical_model():
    assert config_builder.is_technical({"model_name": "technical"})
    assert config_builder.is_technical({"model_type": "technical"})
    assert not config_builder.is_technical({"model_name": "momentum"})
    assert not config_builder.is_technical({"model_name": "reppo-custom"})


def test_build_model_config_technical_maps_levers():
    config = config_builder.build_model_config(
        {
            "model_name": "technical",
            "technical_buy_indicator": "williams10",
            "technical_buy_threshold": -0.5,
            "technical_sell_indicator": "stochasticOscillator10",
            "technical_sell_threshold": 0.6,
        }
    )
    assert config.model_type == "technical"
    assert config.model_technical.buy_indicator == "williams10"
    assert config.model_technical.buy_amount_threshold == -0.5
    assert config.model_technical.sell_indicator == "stochasticOscillator10"
    assert config.model_technical.sell_amount_threshold == 0.6


def test_build_model_config_technical_defaults_to_rsi_mean_reversion():
    config = config_builder.build_model_config({"model_name": "technical"})
    assert config.model_type == "technical"
    # Classic RSI mean-reversion on the [-1,1]-normalised rsi10 (oversold -0.4 / overbought +0.4).
    assert config.model_technical.buy_indicator == "rsi10"
    assert config.model_technical.buy_amount_threshold == -0.4
    assert config.model_technical.sell_indicator == "rsi10"
    assert config.model_technical.sell_amount_threshold == 0.4
    # The default direction flags make it buy-the-dip / sell-the-rip (oversold buy, overbought sell).
    assert config.model_technical.buy_is_down_check is True
    assert config.model_technical.sell_is_up_check is True


def test_build_model_config_technical_defaults_direction_to_mean_reversion():
    config = config_builder.build_model_config({"model_name": "technical"})
    # Default flags = buy oversold / sell overbought (mean-reversion).
    assert config.model_technical.buy_is_down_check is True
    assert config.model_technical.sell_is_up_check is True


def test_build_model_config_technical_direction_flags_enable_trend_following():
    # Trend-following (buy ABOVE / sell BELOW) needs both flags flipped off — the config a moving-average /
    # breakout paper replicates via trendSlope10 crossing 0.
    config = config_builder.build_model_config(
        {
            "model_name": "technical",
            "technical_buy_indicator": "trendSlope10",
            "technical_buy_is_down_check": False,
            "technical_sell_indicator": "trendSlope10",
            "technical_sell_is_up_check": False,
        }
    )
    assert config.model_technical.buy_indicator == "trendSlope10"
    assert config.model_technical.buy_is_down_check is False
    assert config.model_technical.sell_is_up_check is False


def test_build_data_config_technical_forces_use_indicators(monkeypatch):
    _echo_daily(monkeypatch)
    # A technical run needs the curated indicator columns (rsi10, ...) present even if the user left
    # use_indicators off — build_data_config must enable them for the technical model.
    cfg = config_builder.build_data_config({"timeframe": "1d", "model_name": "technical"})
    assert cfg.use_indicators is True


def test_build_data_config_non_technical_keeps_use_indicators_off(monkeypatch):
    _echo_daily(monkeypatch)
    cfg = config_builder.build_data_config({"timeframe": "1d", "model_name": "reppo-custom"})
    assert cfg.use_indicators is False


def test_build_model_config_lstm_levers_default_to_separate_256():
    # Byte-compat default: a reppo-custom run without the new levers keeps SB3's separate-LSTM,
    # hidden-size-256 topology (what the policy used before these levers existed).
    config = config_builder.build_model_config({"model_name": "reppo-custom"})
    assert config.model_rl.lstm_hidden_size == 256
    assert config.model_rl.shared_lstm is False
    assert config.model_rl.enable_critic_lstm is True


def test_build_model_config_lstm_hidden_size_override():
    config = config_builder.build_model_config({"model_name": "reppo-custom", "lstm_hidden_size": 64})
    assert config.model_rl.lstm_hidden_size == 64


def test_build_model_config_shared_lstm_topology():
    config = config_builder.build_model_config(
        {"model_name": "reppo-custom", "shared_lstm": True, "enable_critic_lstm": False}
    )
    assert config.model_rl.shared_lstm is True
    assert config.model_rl.enable_critic_lstm is False


def test_build_model_config_shared_lstm_without_disabling_critic_fails_fast():
    # SB3 asserts shared_lstm XOR critic_lstm; surface that as a clear fail-fast, not a deep assert.
    with pytest.raises(SystemExit):
        config_builder.build_model_config({"model_name": "reppo-custom", "shared_lstm": True})


def test_build_model_config_shared_lstm_ignored_for_non_recurrent_model():
    # A stray shared_lstm on a non-LSTM model is a meaningless lever, not an error.
    config = config_builder.build_model_config({"model_name": "dqn", "shared_lstm": True})
    assert config.model_rl.shared_lstm is True


def test_build_model_config_maps_penalty_multipliers():
    config = config_builder.build_model_config({"combo_noop_penalty": 0.02, "combo_fee_penalty": 2.5})
    assert config.model_rl.reward_multiplier_combo_noop_penalty == 0.02
    assert config.model_rl.reward_multiplier_combo_fee_penalty == 2.5


def test_build_model_config_penalty_multipliers_default():
    # combo_unified defaults BOTH penalties OFF (0) so a bare run ≡ the old combo_all (see build_model_config).
    config = config_builder.build_model_config({})
    assert config.model_rl.reward_multiplier_combo_noop_penalty == 0.0
    assert config.model_rl.reward_multiplier_combo_fee_penalty == 0.0


def test_require_data_present_checks_the_selected_window(monkeypatch):
    seen = set()

    def fake_daily(pairs, symbol=config_builder._SYMBOL):
        seen.update(y for (y, _) in pairs)
        return [f"{y}-{m}" for (y, m) in pairs]

    monkeypatch.setattr(config_builder, "_daily_files", fake_daily)
    config_builder.require_data_present({"walk_forward_window": "2022"})
    assert 2021 in seen
    assert 2022 in seen
    assert 2024 not in seen


def test_require_data_present_raises_when_window_data_missing(monkeypatch):
    monkeypatch.setattr(
        config_builder, "_daily_files", lambda pairs, symbol=config_builder._SYMBOL: []
    )
    with pytest.raises(SystemExit):
        config_builder.require_data_present({"walk_forward_window": "2023"})


# --- _daily_files: builds binance/ paths and filters to those present on disk ---


def test_daily_files_builds_paths_and_filters_to_existing(monkeypatch):
    # Only the second pair's file "exists" -> only its path survives the os.path.exists filter.
    present = "binance/BTCUSDT-1d-2021-2.json"
    monkeypatch.setattr(config_builder.os.path, "exists", lambda p: p == present)
    out = config_builder._daily_files([(2021, 1), (2021, 2)])
    assert out == [present]


def test_daily_files_honours_symbol_and_returns_empty_when_none_present(monkeypatch):
    monkeypatch.setattr(config_builder.os.path, "exists", lambda p: False)
    assert config_builder._daily_files([(2020, 1)], symbol="ETHUSDT") == []
    # And when everything exists, the built path embeds the requested symbol.
    monkeypatch.setattr(config_builder.os.path, "exists", lambda p: True)
    out = config_builder._daily_files([(2020, 3)], symbol="ETHUSDT")
    assert out == ["binance/ETHUSDT-1d-2020-3.json"]


# --- _parse_net_arch: list/tuple coerce to ints vs. comma-string parse ---


def test_parse_net_arch_from_list_and_tuple():
    assert config_builder._parse_net_arch([64, 32]) == [64, 32]
    assert config_builder._parse_net_arch((128, "64")) == [128, 64]


def test_parse_net_arch_from_comma_string_skips_blanks():
    assert config_builder._parse_net_arch("256, 128 ,") == [256, 128]
    assert config_builder._parse_net_arch("") == []


# --- _optional_float: treats None/""/"null"/0 as "unset" (None), else coerces to float ---


@pytest.mark.parametrize("falsy", [None, "", "null", 0])
def test_optional_float_unset_sentinels_become_none(falsy):
    assert config_builder._optional_float({"k": falsy}, "k", "default-unused") is None


def test_optional_float_uses_default_when_key_absent():
    # Default 0.02 (a non-sentinel float) is coerced; default None stays None.
    assert config_builder._optional_float({}, "stop_loss", 0.02) == pytest.approx(0.02)
    assert config_builder._optional_float({}, "take_profit", None) is None


def test_optional_float_coerces_real_value():
    assert config_builder._optional_float({"tp": "0.05"}, "tp", None) == pytest.approx(0.05)


# --- build_env_config: trade_all env with the lever overrides + the optional-float TP/SL fields ---


def test_build_env_config_defaults():
    env = config_builder.build_env_config({})
    assert env.type == "trade_all"
    assert env.initial_balance == 100000
    assert env.transaction_fee == pytest.approx(0.001)
    assert env.take_profit is None
    assert env.trailing_take_profit is None
    assert env.stop_loss == pytest.approx(0.02)
    assert env.no_sell_action is False
    assert env.position_sizing == "fixed"
    assert env.allow_shorting is False
    assert list(env.observations_contain) == [
        "networth_percent_this_trade",
        "in_position",
        "drawdown",
    ]


def test_build_env_config_applies_levers_and_optional_float_zeroing():
    # take_profit set; stop_loss=0 and trailing_take_profit=0 are sentinels -> None (disabled).
    env = config_builder.build_env_config(
        {
            "initial_balance": 50000,
            "transaction_fee": 0.002,
            "take_profit": 0.1,
            "trailing_take_profit": 0,
            "stop_loss": 0,
            "no_sell_action": True,
            "position_sizing": "vol_target",
            "vol_window": 20,
            "allow_shorting": True,
            "max_short_size": 0.5,
        }
    )
    assert env.initial_balance == 50000
    assert env.transaction_fee == pytest.approx(0.002)
    assert env.take_profit == pytest.approx(0.1)
    assert env.trailing_take_profit is None
    assert env.stop_loss is None
    assert env.no_sell_action is True
    assert env.position_sizing == "vol_target"
    assert env.vol_window == 20
    assert env.allow_shorting is True
    assert env.max_short_size == pytest.approx(0.5)


# --- build_model_config: the hodl baseline path ---


def test_build_model_config_hodl_by_model_name():
    config = config_builder.build_model_config({"model_name": "hodl"})
    assert config.model_type == "hodl"
    assert config.iterations_to_pick_best == 1


def test_build_model_config_hodl_by_model_type():
    config = config_builder.build_model_config({"model_type": "hodl"})
    assert config.model_type == "hodl"
    assert config.iterations_to_pick_best == 1


def test_is_hodl_predicate():
    assert config_builder.is_hodl({"model_name": "HODL"})
    assert config_builder.is_hodl({"model_type": "hodl"})
    assert not config_builder.is_hodl({"model_name": "reppo-custom"})


# --- build_model_config (RL): the optional advanced levers each override the tuned default ---


def test_build_model_config_non_custom_model_clears_custom_net_arch_tokens():
    # The tuned custom_net_arch tokens only apply to *-custom models; a plain model name clears them.
    custom = config_builder.build_model_config({"model_name": "reppo-custom"})
    plain = config_builder.build_model_config({"model_name": "ppo"})
    assert custom.model_rl.custom_net_arch  # tuned tokens retained
    assert plain.model_rl.custom_net_arch == []


def test_build_model_config_rl_optional_levers_applied():
    config = config_builder.build_model_config(
        {
            "model_name": "ppo",
            "checkpoint_to_load": "checkpoints/prev.zip",
            "net_arch": "64,64",
            "optimizer_class": "AdamW",
            "activation_fn": "relu",
            "exploration_fraction": 0.3,
            "exploration_final_eps": 0.05,
        }
    )
    rl = config.model_rl
    assert rl.checkpoint_to_load == "checkpoints/prev.zip"
    assert rl.net_arch == [64, 64]
    assert rl.optimizer_class == "AdamW"
    assert rl.activation_fn == "relu"
    assert rl.exploration_fraction == pytest.approx(0.3)
    assert rl.exploration_final_eps == pytest.approx(0.05)
    assert config.iterations_to_pick_best == 1


def test_build_model_config_rl_net_arch_accepts_list():
    config = config_builder.build_model_config({"model_name": "ppo", "net_arch": [128, 32]})
    assert config.model_rl.net_arch == [128, 32]


def test_build_model_config_rl_seed_passed_through_and_none_when_absent():
    seeded = config_builder.build_model_config({"model_name": "ppo", "seed": 11})
    assert seeded.model_rl.seed == 11
    unseeded = config_builder.build_model_config({"model_name": "ppo"})
    assert unseeded.model_rl.seed is None


# --- build_data_config: intraday on a non-BTC asset is rejected (no intraday altcoin klines) ---


def test_build_data_config_intraday_non_btc_fails_fast():
    with pytest.raises(SystemExit, match="no intraday dataset"):
        config_builder.build_data_config({"timeframe": "1h", "asset": "ETHUSDT"})
