"""Prove BlackSwan training runs on Apple Silicon's Metal (MPS) backend end-to-end.

These are real training runs, not mocks: each test builds the actual (data, env, model)
from a flat lever config and drives it through ``trainer.run._run_one`` on ``device="mps"``,
exactly as the Model Trainer would, then asserts the model's parameters actually live on the
MPS device (so a silent CPU fallback fails the test) and that a valid RunSummary comes out.

They are gated to skip cleanly when MPS isn't present (non-Apple-Silicon hosts, CUDA/CI boxes)
or when the 1d BTCUSDT klines aren't on disk, so the suite stays green everywhere while still
giving a runnable proof on a Mac with Metal. 1d data is used because it is the fastest path
(one bar per day -> a whole window trains in seconds).

Run just these:  pytest trainer/test_mps_integration.py -s
Include the slower default production model (reppo-custom, an LSTM):  BS_MPS_FULL=1 pytest trainer/test_mps_integration.py -s
"""

import math
import os

import pytest

try:
    import torch

    _MPS_OK = torch.backends.mps.is_available()
except Exception:
    torch = None
    _MPS_OK = False

from trainer import config_builder
from trainer import run as trainer_run
from trainer import summary as summary_mod

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

requires_mps = pytest.mark.skipif(not _MPS_OK, reason="torch MPS backend not available on this host")


def _data_present(cfg):
    try:
        config_builder.require_data_present(cfg)
        return True
    except SystemExit:
        return False


def _fast_cfg(model_name):
    """A deliberately tiny 1d trading run: small net, short buffer, a single narrow window."""
    return {
        "model_name": model_name,
        "reward_model": "combo_all2",
        "learning_rate": 0.0001,
        "gamma": 0.99,
        "batch_size": 64,
        "buffer_size": 5000,
        "learning_starts": 50,
        "net_arch": "64,64",
        "episodes": 1,
        "timeframe": "1d",
        "walk_forward_window": "2022",
        "device": "mps",
        "seed": 0,
    }


@pytest.fixture(autouse=True)
def _chdir_repo_root(monkeypatch):
    # _run_one / require_data_present resolve binance/ + checkpoints/ relative to the repo root.
    monkeypatch.chdir(_REPO_ROOT)


@requires_mps
def test_torch_mps_backend_is_available_and_built():
    assert torch.backends.mps.is_built()
    assert torch.backends.mps.is_available()


@requires_mps
def test_mps_compute_matches_cpu():
    """A real op executes on the device and agrees with CPU — guards against a silent no-op backend."""
    torch.manual_seed(0)
    a = torch.randn(128, 128)
    b = torch.randn(128, 128)
    cpu = a @ b
    mps = (a.to("mps") @ b.to("mps")).cpu()
    assert mps.shape == cpu.shape
    assert torch.allclose(mps, cpu, atol=1e-3)


@requires_mps
@pytest.mark.skipif(
    os.environ.get("BS_MPS") != "1",
    reason="real end-to-end training run (~30s each); set BS_MPS=1 to include the MPS pipeline smoke test",
)
@pytest.mark.parametrize("model_name", ["dqn", "ppo"])
def test_simple_model_trains_end_to_end_on_mps(model_name):
    cfg = _fast_cfg(model_name)
    if not _data_present(cfg):
        pytest.skip("1d BTCUSDT klines not on disk")

    env_test, state, model, is_rl, train_seconds = trainer_run._run_one(cfg)

    assert model.rl_model.policy.device.type == "mps"
    assert is_rl is True
    assert train_seconds > 0
    assert state is not None

    out = summary_mod.build_summary(
        env_test, state, cfg, model, "2024-01-01T00:00:00+00:00", is_rl
    )
    assert isinstance(out["objective"], float) and math.isfinite(out["objective"])
    assert out["health"]["status"] in {"ok", "degenerate", "warn", "error"}
    assert out["config"]["device"] == "mps"


@requires_mps
@pytest.mark.skipif(
    os.environ.get("BS_MPS_FULL") != "1",
    reason="slow LSTM run; set BS_MPS_FULL=1 to include the default production model",
)
def test_default_production_model_reppo_custom_trains_on_mps():
    cfg = _fast_cfg("reppo-custom")
    if not _data_present(cfg):
        pytest.skip("1d BTCUSDT klines not on disk")

    env_test, state, model, is_rl, train_seconds = trainer_run._run_one(cfg)

    assert model.rl_model.policy.device.type == "mps"
    out = summary_mod.build_summary(
        env_test, state, cfg, model, "2024-01-01T00:00:00+00:00", is_rl
    )
    assert math.isfinite(out["objective"])
