"""Unit tests for trainer/run.py — the trainer-conformant single-config CLI.

The heavy collaborators (data/env/model factories + the real train loop) are
monkeypatched out so only run.py's PURE decision logic is exercised: the
@@PROGRESS marker, RNG seeding, JSON write, and — the bulk — main()'s argument
parsing + branch selection (calibrate / config-json / resume / evaluate). The
actual `_run_one` train/test pipeline is replaced by a stub in main() tests
(see notes for what is deliberately not unit-tested).
"""

import json
import random

import numpy as np
import pytest

from trainer import run as run_mod


# --- _CALIBRATE_CFG ----------------------------------------------------------


def test_calibrate_cfg_is_a_tiny_one_episode_cpu_dqn():
    cfg = run_mod._CALIBRATE_CFG
    assert cfg["model_name"] == "dqn"
    assert cfg["episodes"] == 1
    assert cfg["timeframe"] == "1d"
    assert cfg["device"] == "cpu"


# --- _progress ---------------------------------------------------------------


def test_progress_emits_parseable_marker(capsys):
    run_mod._progress("loading")
    line = capsys.readouterr().out.strip()
    assert line.startswith("@@PROGRESS ")
    payload = json.loads(line[len("@@PROGRESS "):])
    assert payload == {"phase": "loading"}


def test_progress_merges_extra_fields(capsys):
    run_mod._progress("train", total=500, foo="bar")
    line = capsys.readouterr().out.strip()
    payload = json.loads(line[len("@@PROGRESS "):])
    assert payload == {"phase": "train", "total": 500, "foo": "bar"}


# --- _seed_everything --------------------------------------------------------


def test_seed_everything_makes_python_random_deterministic():
    run_mod._seed_everything(123)
    a = [random.random() for _ in range(3)]
    run_mod._seed_everything(123)
    b = [random.random() for _ in range(3)]
    assert a == b


def test_seed_everything_makes_numpy_deterministic():
    run_mod._seed_everything(7)
    a = np.random.rand(4).tolist()
    run_mod._seed_everything(7)
    b = np.random.rand(4).tolist()
    assert a == b


def test_seed_everything_different_seeds_differ():
    run_mod._seed_everything(1)
    a = np.random.rand(4).tolist()
    run_mod._seed_everything(2)
    b = np.random.rand(4).tolist()
    assert a != b


# --- _write ------------------------------------------------------------------


def test_write_round_trips_json(tmp_path):
    path = str(tmp_path / "out.json")
    payload = {"objective": 1.5, "nested": {"a": [1, 2, 3]}}
    run_mod._write(path, payload)
    with open(path) as f:
        assert json.load(f) == payload


# --- main(): argument-required branches --------------------------------------


def test_main_requires_summary_out():
    # --summary-out is required; argparse exits non-zero.
    with pytest.raises(SystemExit):
        run_mod.main(["--calibrate"])


def test_main_requires_config_or_calibrate(tmp_path):
    # Neither --config-json nor --calibrate → parser.error → SystemExit.
    with pytest.raises(SystemExit):
        run_mod.main(["--summary-out", str(tmp_path / "s.json")])


# --- main(): a stubbed pipeline so the branch logic runs end-to-end ----------


def _patch_pipeline(monkeypatch, *, captured, train_seconds=2.0, objective=0.5):
    """Replace run.py's heavy collaborators with stubs that record the cfg main() assembled.

    `captured["cfg"]` is the cfg main() handed to `_run_one`; `captured["summary_cfg"]`
    is what reached build_summary; trace/decision calls are recorded so the calibrate-skips-
    trace and evaluate-stamps-evaluation branches can be asserted.
    """
    captured.setdefault("trace_called", 0)

    def fake_run_one(cfg):
        captured["cfg"] = cfg
        # (env_test, run_state, model, is_rl, train_seconds)
        return object(), object(), object(), True, train_seconds

    def fake_build_summary(env_test, state, cfg, model, ran_at, is_rl):
        captured["summary_cfg"] = cfg
        captured["ran_at"] = ran_at
        captured["is_rl"] = is_rl
        return {"objective": objective, "health": {"status": "ok"}, "artifacts": {}}

    def fake_attach(out, env_test, model, cfg, summary_out, is_rl):
        captured["trace_called"] += 1

    monkeypatch.setattr(run_mod, "_run_one", fake_run_one)
    monkeypatch.setattr(run_mod.summary_mod, "build_summary", fake_build_summary)
    monkeypatch.setattr(run_mod.decision_trace, "attach_decision_trace", fake_attach)
    # main() chdir's to the repo root; neutralise it so tmp paths + cwd stay stable.
    monkeypatch.setattr(run_mod.os, "chdir", lambda *a, **k: None)


def test_main_calibrate_uses_builtin_cfg_and_skips_trace(tmp_path, monkeypatch):
    captured = {}
    _patch_pipeline(monkeypatch, captured=captured, train_seconds=2.0)
    out_path = str(tmp_path / "s.json")
    rc = run_mod.main(["--calibrate", "--summary-out", out_path])
    assert rc == 0
    # calibrate sources cfg from the builtin template (a COPY, not the module constant).
    assert captured["cfg"]["model_name"] == "dqn"
    assert captured["cfg"] is not run_mod._CALIBRATE_CFG
    # decision-trace is intentionally skipped for calibrate runs.
    assert captured["trace_called"] == 0


def test_main_calibrate_stamps_calibration_block(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch, captured={}, train_seconds=4.0)
    out_path = str(tmp_path / "s.json")
    run_mod.main(["--calibrate", "--summary-out", out_path])
    with open(out_path) as f:
        out = json.load(f)
    cal = out["calibration"]
    assert cal["units"] == 1
    assert cal["secondsObserved"] == pytest.approx(4.0)
    # 1 episode over 4 seconds → 0.25 units/sec.
    assert cal["unitsPerSecond"] == pytest.approx(0.25)


def test_main_calibrate_zero_train_seconds_yields_zero_rate(tmp_path, monkeypatch):
    # Guard against div-by-zero: a 0s observation must report 0.0, not raise.
    _patch_pipeline(monkeypatch, captured={}, train_seconds=0.0)
    out_path = str(tmp_path / "s.json")
    run_mod.main(["--calibrate", "--summary-out", out_path])
    with open(out_path) as f:
        out = json.load(f)
    assert out["calibration"]["unitsPerSecond"] == 0.0


def test_main_config_json_loads_cfg_from_file(tmp_path, monkeypatch):
    captured = {}
    _patch_pipeline(monkeypatch, captured=captured)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"model_name": "reppo-custom", "seed": 5}))
    out_path = str(tmp_path / "s.json")
    rc = run_mod.main(["--config-json", str(cfg_path), "--summary-out", out_path])
    assert rc == 0
    assert captured["cfg"]["model_name"] == "reppo-custom"
    assert captured["cfg"]["seed"] == 5
    # non-calibrate runs DO attach a decision trace.
    assert captured["trace_called"] == 1


def test_main_config_json_no_calibration_block(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch, captured={})
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"model_name": "dqn"}))
    out_path = str(tmp_path / "s.json")
    run_mod.main(["--config-json", str(cfg_path), "--summary-out", out_path])
    with open(out_path) as f:
        out = json.load(f)
    assert "calibration" not in out
    assert "evaluation" not in out


def test_resolve_device_explicit_value_wins_over_auto_logic():
    assert run_mod._resolve_device({"device": "cpu"}) == "cpu"
    assert run_mod._resolve_device({"device": "mps", "model_name": "ppo"}) == "mps"
    assert run_mod._resolve_device({}) == "cpu"  # default


def test_resolve_device_auto_never_picks_mps_for_trading_models(monkeypatch):
    # MPS is measurably SLOWER for the trading line's small-net, single-env models (bench_mps.py:
    # dqn 18.6s cpu vs 44.9s mps) and the LSTM models additionally hit an intermittent Metal
    # LSTM-gradient assertion (GPURNNOps.mm) that aborts training mid-run. So `auto` stays on CPU;
    # only an explicit device="mps" forces Metal.
    monkeypatch.setattr(run_mod, "_mps_available", lambda: True)
    for name in ("reppo", "reppo-custom", "dqn", "duel-dqn-custom-lstm"):
        assert run_mod._resolve_device({"device": "auto", "model_name": name}) == "cpu"


def test_resolve_device_auto_keeps_cpu_for_mlp_models(monkeypatch):
    monkeypatch.setattr(run_mod, "_mps_available", lambda: True)
    for name in ("ppo", "a2c", "ars", ""):
        assert run_mod._resolve_device({"device": "auto", "model_name": name}) == "cpu"


def test_resolve_device_auto_falls_back_to_cpu_without_mps(monkeypatch):
    monkeypatch.setattr(run_mod, "_mps_available", lambda: False)
    assert run_mod._resolve_device({"device": "auto", "model_name": "reppo-custom"}) == "cpu"


def test_main_emit_decision_trace_false_skips_trace(tmp_path, monkeypatch):
    # A sweep run can opt OUT of the (expensive) second deterministic test replay that builds the
    # decision trace by setting emit_decision_trace=false — the scored summary is unaffected.
    captured = {}
    _patch_pipeline(monkeypatch, captured=captured)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"model_name": "dqn", "emit_decision_trace": False}))
    out_path = str(tmp_path / "s.json")
    rc = run_mod.main(["--config-json", str(cfg_path), "--summary-out", out_path])
    assert rc == 0
    assert captured["trace_called"] == 0


def test_main_emit_decision_trace_defaults_on(tmp_path, monkeypatch):
    # Omitting the flag preserves the historical behaviour: the trace IS attached.
    captured = {}
    _patch_pipeline(monkeypatch, captured=captured)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"model_name": "dqn"}))
    out_path = str(tmp_path / "s.json")
    run_mod.main(["--config-json", str(cfg_path), "--summary-out", out_path])
    assert captured["trace_called"] == 1


def test_main_resume_from_sets_checkpoint_to_load(tmp_path, monkeypatch):
    captured = {}
    _patch_pipeline(monkeypatch, captured=captured)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"model_name": "dqn"}))
    out_path = str(tmp_path / "s.json")
    run_mod.main(
        ["--config-json", str(cfg_path), "--summary-out", out_path, "--resume-from", "myckpt"]
    )
    assert captured["cfg"]["checkpoint_to_load"] == "myckpt"


def test_main_decision_trace_failure_does_not_fail_run(tmp_path, monkeypatch, capsys):
    captured = {}
    _patch_pipeline(monkeypatch, captured=captured)

    def boom(*a, **k):
        raise RuntimeError("trace exploded")

    monkeypatch.setattr(run_mod.decision_trace, "attach_decision_trace", boom)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"model_name": "dqn"}))
    out_path = str(tmp_path / "s.json")
    # A decision-trace failure must NOT fail an otherwise-good run.
    rc = run_mod.main(["--config-json", str(cfg_path), "--summary-out", out_path])
    assert rc == 0
    assert "decision-trace skipped" in capsys.readouterr().out


# --- main(): --evaluate branch ----------------------------------------------


def test_main_evaluate_requires_a_checkpoint(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch, captured={})
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"model_name": "dqn"}))
    out_path = str(tmp_path / "s.json")
    # No checkpoint anywhere → parser.error → SystemExit.
    with pytest.raises(SystemExit):
        run_mod.main(["--config-json", str(cfg_path), "--summary-out", out_path, "--evaluate"])


def test_main_evaluate_reduces_artifact_path_to_bare_id(tmp_path, monkeypatch):
    captured = {}
    _patch_pipeline(monkeypatch, captured=captured)
    cfg_path = tmp_path / "cfg.json"
    # the hub injects the run's `checkpoint` artifact as "<folder>/<id>.zip".
    cfg_path.write_text(json.dumps({"model_name": "dqn", "checkpoint": "checkpoints/run-abc.zip"}))
    out_path = str(tmp_path / "s.json")
    rc = run_mod.main(
        ["--config-json", str(cfg_path), "--summary-out", out_path, "--evaluate"]
    )
    assert rc == 0
    # the loader re-adds folder + .zip, so cfg gets the bare id.
    assert captured["cfg"]["checkpoint_to_load"] == "run-abc"


def test_main_evaluate_accepts_checkpoint_to_load_key(tmp_path, monkeypatch):
    captured = {}
    _patch_pipeline(monkeypatch, captured=captured)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"model_name": "dqn", "checkpoint_to_load": "bare-id"}))
    out_path = str(tmp_path / "s.json")
    run_mod.main(["--config-json", str(cfg_path), "--summary-out", out_path, "--evaluate"])
    # bare id with no .zip / folder is left untouched.
    assert captured["cfg"]["checkpoint_to_load"] == "bare-id"


def test_main_evaluate_strips_zip_but_keeps_non_zip_basename(tmp_path, monkeypatch):
    captured = {}
    _patch_pipeline(monkeypatch, captured=captured)
    cfg_path = tmp_path / "cfg.json"
    # a path without .zip: only the directory is stripped, the suffix-less basename stays.
    cfg_path.write_text(json.dumps({"model_name": "dqn", "checkpoint": "ckpts/run-xyz"}))
    out_path = str(tmp_path / "s.json")
    run_mod.main(["--config-json", str(cfg_path), "--summary-out", out_path, "--evaluate"])
    assert captured["cfg"]["checkpoint_to_load"] == "run-xyz"


def test_main_evaluate_stamps_evaluation_block(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch, captured={})
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"model_name": "dqn", "checkpoint": "checkpoints/run-abc.zip"}))
    out_path = str(tmp_path / "s.json")
    run_mod.main(["--config-json", str(cfg_path), "--summary-out", out_path, "--evaluate"])
    with open(out_path) as f:
        out = json.load(f)
    assert out["evaluation"] == {"checkpoint": "run-abc", "episodes": 0}


def test_main_resume_and_evaluate_compose(tmp_path, monkeypatch):
    # --resume-from sets checkpoint_to_load first; --evaluate then reduces it (no folder/.zip → unchanged).
    captured = {}
    _patch_pipeline(monkeypatch, captured=captured)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"model_name": "dqn"}))
    out_path = str(tmp_path / "s.json")
    run_mod.main(
        [
            "--config-json",
            str(cfg_path),
            "--summary-out",
            out_path,
            "--resume-from",
            "checkpoints/saved.zip",
            "--evaluate",
        ]
    )
    assert captured["cfg"]["checkpoint_to_load"] == "saved"


# --- main(): output side-effects --------------------------------------------


def test_main_writes_summary_and_prints_objective(tmp_path, monkeypatch, capsys):
    _patch_pipeline(monkeypatch, captured={}, objective=0.4242)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"model_name": "dqn"}))
    out_path = str(tmp_path / "s.json")
    rc = run_mod.main(["--config-json", str(cfg_path), "--summary-out", out_path])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "objective(total_return_pct)=0.4242" in printed
    assert "status=ok" in printed
    assert out_path in printed


def test_main_passes_is_rl_through_to_summary(tmp_path, monkeypatch):
    captured = {}
    _patch_pipeline(monkeypatch, captured=captured)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"model_name": "dqn"}))
    out_path = str(tmp_path / "s.json")
    run_mod.main(["--config-json", str(cfg_path), "--summary-out", out_path])
    # _run_one's is_rl flag flows into build_summary unchanged.
    assert captured["is_rl"] is True


# --- A6: mid-training snapshot traces ----------------------------------------


def test_trace_checkpoint_loads_snapshot_and_returns_its_trace(monkeypatch):
    captured = {}

    def fake_run_one(cfg):
        captured["cfg"] = dict(cfg)
        return object(), object(), object(), True, 1.0

    def fake_attach(out, env_test, model, cfg, summary_out, is_rl):
        out.setdefault("artifacts", {})["decisionTrace"] = {
            "steps": [{"step": 0, "action": "hold"}],
            "actionCounts": {"hold": 1},
            "totalSteps": 1,
        }

    monkeypatch.setattr(run_mod, "_run_one", fake_run_one)
    monkeypatch.setattr(run_mod.decision_trace, "attach_decision_trace", fake_attach)
    cfg = {"model_name": "dqn"}
    trace = run_mod._trace_checkpoint(cfg, "m.step100", cheap=True)
    assert trace["totalSteps"] == 1
    # loaded via checkpoint_to_load; cheap disables the expensive replays
    assert captured["cfg"]["checkpoint_to_load"] == "m.step100"
    assert captured["cfg"]["decision_trace_attribution"] is False
    assert captured["cfg"]["decision_trace_attention"] is False
    assert captured["cfg"]["decision_trace_latent"] is False
    assert cfg == {"model_name": "dqn"}  # original cfg not mutated


def test_trace_checkpoint_returns_none_when_no_trace(monkeypatch):
    monkeypatch.setattr(run_mod, "_run_one", lambda cfg: (object(), object(), object(), True, 1.0))
    monkeypatch.setattr(run_mod.decision_trace, "attach_decision_trace", lambda *a, **k: None)
    assert run_mod._trace_checkpoint({"model_name": "dqn"}, "m.step100") is None


def test_main_writes_snapshot_traces_from_model_snapshots(tmp_path, monkeypatch):
    class _M:
        id = "runid"
        snapshots = [{"step": 100, "path": "checkpoints/runid.step100"}]

        def produces_checkpoint(self):
            return True

    monkeypatch.setattr(run_mod, "_run_one", lambda cfg: (object(), object(), _M(), True, 1.0))
    monkeypatch.setattr(
        run_mod.summary_mod,
        "build_summary",
        lambda *a, **k: {"objective": 0.5, "health": {"status": "ok"}, "artifacts": {}},
    )
    monkeypatch.setattr(run_mod.decision_trace, "attach_decision_trace", lambda *a, **k: None)
    monkeypatch.setattr(
        run_mod,
        "_trace_checkpoint",
        lambda cfg, ref, cheap=True: {"steps": [{"step": 0, "action": "hold"}], "actionCounts": {"hold": 1}, "totalSteps": 1},
    )
    monkeypatch.chdir(tmp_path)  # real chdir FIRST, then neutralise main()'s chdir-to-repo-root
    monkeypatch.setattr(run_mod.os, "chdir", lambda *a, **k: None)
    (tmp_path / "checkpoints").mkdir()
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"model_name": "dqn"}))
    out_path = str(tmp_path / "s.json")
    assert run_mod.main(["--config-json", str(cfg_path), "--summary-out", out_path]) == 0
    out = json.loads((tmp_path / "s.json").read_text())
    idx = out["artifacts"]["snapshotTraces"]
    assert [e["step"] for e in idx] == [100]
    assert idx[0]["traceFile"] == "checkpoints/runid.snapshots.jsonl"
    assert (tmp_path / "checkpoints" / "runid.snapshots.jsonl").exists()
