import json
import os

import pytest

from trainer import run_dip


# --- tiny fakes ---------------------------------------------------------------


class _FakeEnv:
    """The dip test env surface build_summary reads."""

    def __init__(self, n_positive=5, n_negative=5, predictions=None):
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.predictions = predictions if predictions is not None else [0, 1, 0, 1]


class _FakeModel:
    def __init__(self, model_id="ck-1"):
        self.id = model_id


def _ok_state():
    # [f1, simple_ratio, accuracy, precision, recall, negative_recall]
    return [0.7, 0.5, 0.65, 0.6, 0.6, 0.4]


def _patch_run_one(monkeypatch, *, state=None, env=None, model=None, train_seconds=2.0, recorder=None):
    """Replace the heavy build->train->test pipeline with a deterministic stub.

    `_run_one` builds providers/envs/models and runs a real (tiny) training loop, which needs
    on-disk klines + torch; out of scope for a fast unit test, so we stub it and assert only the
    CLI orchestration around it.
    """
    env = env if env is not None else _FakeEnv()
    model = model if model is not None else _FakeModel()
    state = state if state is not None else _ok_state()

    def fake_run_one(cfg):
        if recorder is not None:
            recorder.append(dict(cfg))
        return env, state, model, train_seconds

    monkeypatch.setattr(run_dip, "_run_one", fake_run_one)
    return env, state, model


# --- argument parsing ---------------------------------------------------------


def test_main_requires_summary_out(monkeypatch):
    _patch_run_one(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        run_dip.main(["--calibrate"])
    assert exc.value.code == 2


def test_main_errors_without_config_or_calibrate(monkeypatch, tmp_path):
    _patch_run_one(monkeypatch)
    out = str(tmp_path / "s.json")
    with pytest.raises(SystemExit) as exc:
        run_dip.main(["--summary-out", out])
    assert exc.value.code == 2


# --- config-json path ---------------------------------------------------------


def test_main_loads_config_json_and_writes_summary(monkeypatch, tmp_path):
    seen = []
    _patch_run_one(monkeypatch, recorder=seen)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"asset": "ETHUSDT", "episodes": 3}))
    out_path = tmp_path / "sum.json"

    rc = run_dip.main(["--config-json", str(cfg_path), "--summary-out", str(out_path)])
    assert rc == 0
    # the loaded config is the one fed to the pipeline.
    assert seen == [{"asset": "ETHUSDT", "episodes": 3}]
    out = json.loads(out_path.read_text())
    assert out["objective"] == pytest.approx(0.7)
    assert out["dataset"]["asset"] == "ETHUSDT"
    # a plain (non-calibrate) run carries no calibration block.
    assert "calibration" not in out


def test_main_writes_to_the_requested_summary_path(monkeypatch, tmp_path):
    _patch_run_one(monkeypatch)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({}))
    out_path = tmp_path / "nested" / "out.json"
    out_path.parent.mkdir()

    run_dip.main(["--config-json", str(cfg_path), "--summary-out", str(out_path)])
    assert out_path.exists()
    assert "objective" in json.loads(out_path.read_text())


# --- calibrate path -----------------------------------------------------------


def test_main_calibrate_uses_builtin_config(monkeypatch, tmp_path):
    seen = []
    _patch_run_one(monkeypatch, recorder=seen, train_seconds=4.0)
    out_path = tmp_path / "cal.json"

    rc = run_dip.main(["--calibrate", "--summary-out", str(out_path)])
    assert rc == 0
    # the pipeline gets a COPY of the built-in calibrate config (loss_fn/episodes/etc).
    assert seen[0] == run_dip._CALIBRATE_CFG
    assert seen[0] is not run_dip._CALIBRATE_CFG


def test_main_calibrate_emits_units_per_second(monkeypatch, tmp_path):
    _patch_run_one(monkeypatch, train_seconds=4.0)
    out_path = tmp_path / "cal.json"
    run_dip.main(["--calibrate", "--summary-out", str(out_path)])
    cal = json.loads(out_path.read_text())["calibration"]
    # the calibrate config runs episodes=1 over 4 observed seconds.
    assert cal["units"] == 1
    assert cal["secondsObserved"] == pytest.approx(4.0)
    assert cal["unitsPerSecond"] == pytest.approx(0.25)


def test_main_calibrate_zero_seconds_yields_zero_rate(monkeypatch, tmp_path):
    # a sub-clock-tick train must not divide by zero.
    _patch_run_one(monkeypatch, train_seconds=0.0)
    out_path = tmp_path / "cal.json"
    run_dip.main(["--calibrate", "--summary-out", str(out_path)])
    cal = json.loads(out_path.read_text())["calibration"]
    assert cal["unitsPerSecond"] == 0.0
    assert cal["secondsObserved"] == 0.0


# --- resume-from --------------------------------------------------------------


def test_main_resume_from_injects_checkpoint_to_load(monkeypatch, tmp_path):
    seen = []
    _patch_run_one(monkeypatch, recorder=seen)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"asset": "BTCUSDT"}))
    out_path = tmp_path / "sum.json"

    run_dip.main(
        ["--config-json", str(cfg_path), "--summary-out", str(out_path), "--resume-from", "run-99"]
    )
    assert seen[0]["checkpoint_to_load"] == "run-99"


def test_main_resume_from_overrides_calibrate_config(monkeypatch, tmp_path):
    seen = []
    _patch_run_one(monkeypatch, recorder=seen)
    out_path = tmp_path / "cal.json"
    run_dip.main(["--calibrate", "--summary-out", str(out_path), "--resume-from", "warm"])
    assert seen[0]["checkpoint_to_load"] == "warm"


# --- summary contents are produced via dip.build_summary ----------------------


def test_main_summary_reflects_state_and_health(monkeypatch, tmp_path):
    # a degenerate (no positive signal) run state surfaces in the written health block.
    _patch_run_one(monkeypatch, state=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({}))
    out_path = tmp_path / "sum.json"
    run_dip.main(["--config-json", str(cfg_path), "--summary-out", str(out_path)])
    out = json.loads(out_path.read_text())
    assert out["objective"] == 0.0
    assert out["health"]["status"] == "degenerate"


def test_main_includes_checkpoint_artifact_when_model_has_id(monkeypatch, tmp_path):
    _patch_run_one(monkeypatch, model=_FakeModel("dip-42"))
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({}))
    out_path = tmp_path / "sum.json"
    run_dip.main(["--config-json", str(cfg_path), "--summary-out", str(out_path)])
    out = json.loads(out_path.read_text())
    assert out["artifacts"]["checkpoint"] == "checkpoints/dip-42"


def test_main_seed_propagated_into_summary(monkeypatch, tmp_path):
    seen = []
    _patch_run_one(monkeypatch, recorder=seen)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"seed": 13}))
    out_path = tmp_path / "sum.json"
    run_dip.main(["--config-json", str(cfg_path), "--summary-out", str(out_path)])
    out = json.loads(out_path.read_text())
    assert out["seed"] == 13
    assert out["provenance"]["seed"] == 13


# --- module-level constant ----------------------------------------------------


def test_calibrate_cfg_is_a_short_1d_run():
    cfg = run_dip._CALIBRATE_CFG
    assert cfg["episodes"] == 1
    assert cfg["device"] == "cpu"
    assert cfg["loss_fn"] == "bcelogits"


def test_run_dip_reuses_trading_line_write_helper():
    # the dip CLI must reuse the trading line's write/seed/progress helpers, not fork them.
    from trainer import run as trading_run

    assert run_dip._write is trading_run._write
