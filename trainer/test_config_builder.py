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


def test_build_data_config_incoherent_timeframe_fidelity_fails_fast():
    # Daily step + a finer/multi stack is not provider-supported — must fail fast, not silently run.
    with pytest.raises(SystemExit):
        config_builder.build_data_config({"timeframe": "1d", "fidelity_set": "1h+1d"})


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
