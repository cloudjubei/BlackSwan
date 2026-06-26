"""Direct tests for the pure helpers in trainer/bench_device.py (the heavy main() is a benchmark run)."""

from trainer.bench_device import build_device_summary, pick_best_device


def test_pick_best_device_picks_the_faster_device():
    best, speedup = pick_best_device({"cpu": 60.0, "mps": 40.0})
    assert best == "mps"
    assert speedup == 1.5


def test_pick_best_device_prefers_cpu_on_a_near_tie():
    # MPS must be MEANINGFULLY faster (>=5%) to win — a near-tie keeps CPU (reproducible, no GPU contention).
    assert pick_best_device({"cpu": 100.0, "mps": 98.0}) == ("cpu", 1.0)
    assert pick_best_device({"cpu": 100.0, "mps": 90.0})[0] == "mps"


def test_pick_best_device_cpu_only_when_mps_absent():
    assert pick_best_device({"cpu": 12.3}) == ("cpu", 1.0)


def test_pick_best_device_ignores_missing_or_nonpositive_times():
    assert pick_best_device({"cpu": 10.0, "mps": 0.0}) == ("cpu", 1.0)
    assert pick_best_device({"cpu": None, "mps": None}) == ("cpu", 1.0)


def test_build_device_summary_shape_and_winner():
    out = build_device_summary("reppo-custom", {"cpu": 60.0, "mps": 40.0}, budget=2000)
    db = out["deviceBenchmark"]
    assert db["modelName"] == "reppo-custom"
    assert db["bestDevice"] == "mps"
    assert db["speedup"] == 1.5
    assert db["availableDevices"] == ["cpu", "mps"]
    assert db["usPerStep"] == {"cpu": 30000.0, "mps": 20000.0}  # 1e6*sec/budget


def test_build_device_summary_carries_errors_and_cpu_only():
    out = build_device_summary("dqn-sbx", {"cpu": 5.0}, budget=1000, errors={"mps": "RuntimeError: x"})
    db = out["deviceBenchmark"]
    assert db["bestDevice"] == "cpu"
    assert db["availableDevices"] == ["cpu"]
    assert db["errors"] == {"mps": "RuntimeError: x"}
