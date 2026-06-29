"""Benchmark ONE model on CPU vs MPS and report which device trains it faster.

The Model Trainer's per-model "Benchmark device" button runs this through the trainer's compute runner
(same {summaryOut} contract as --calibrate). The model to benchmark is named by the BENCH_MODEL_NAME env
var (the tool sets it from the model record); BENCH_BUDGET / BENCH_WARMUP / BENCH_DEVICE_TIMEOUT / BENCH_WINDOW
tune the probe. It trains a short fixed step budget on each AVAILABLE device (cpu always; mps when torch
reports it) under a per-device wall-clock cap, times it, and writes the comparison + the winning device to
the summary so the tool can persist `preferredDevice`.

  BENCH_MODEL_NAME=reppo-custom .venv/bin/python -m trainer.bench_device --summary-out out.json

Pure helpers (pick_best_device / build_device_summary) are unit-tested; main() does the heavy run.
"""

import argparse
import json
import os
import signal
import sys
import time


def pick_best_device(timings):
    """Given {device: seconds_for_budget} (lower = faster), return (best_device, speedup_vs_next).

    speedup = the runner-up's time / the winner's time (>= 1.0; 1.0 when only one device ran). Ties and
    non-positive/missing times resolve to 'cpu' so the result is always a usable device."""
    valid = {d: t for d, t in timings.items() if isinstance(t, (int, float)) and t > 0}
    if not valid:
        return "cpu", 1.0
    best = min(valid, key=valid.get)
    others = [t for d, t in valid.items() if d != best]
    speedup = (min(others) / valid[best]) if others else 1.0
    # Prefer cpu on a tie (reproducible + no GPU contention) — only pick mps when it's meaningfully faster.
    if best == "mps" and speedup < 1.05:
        return "cpu", 1.0
    return best, round(speedup, 3)


def build_device_summary(model_name, timings, budget, errors=None):
    """Assemble the {summaryOut} payload the benchmarkModelDevice tool reads back."""
    best, speedup = pick_best_device(timings)
    us_per_step = {d: round(1e6 * t / budget, 1) for d, t in timings.items() if t and t > 0}
    return {
        "deviceBenchmark": {
            "modelName": model_name,
            "budget": budget,
            "seconds": {d: round(t, 3) for d, t in timings.items() if t and t > 0},
            "usPerStep": us_per_step,
            "bestDevice": best,
            "speedup": speedup,
            "availableDevices": [d for d, t in timings.items() if t and t > 0],
            **({"errors": errors} if errors else {}),
        }
    }


def _available_devices():
    devices = ["cpu"]
    try:
        import torch

        if torch.backends.mps.is_available():
            devices.append("mps")
    except Exception:
        pass
    return devices


def _time_device(model_name, device, budget, warmup, prov, env_cfg):
    """Train `budget` steps on `device`, returning elapsed seconds (raises on a device that can't run it)."""
    import torch

    from trainer import config_builder
    from src.environment.env_factory import create_environment
    from src.model.model_factory import create_model

    cfg = {"asset": "BTCUSDT", "timeframe": "1h", "data_type": "only_price_percent",
           "use_indicators": True, "model_name": model_name, "device": device,
           "learning_starts": 200, "episodes": 1}
    env = create_environment(env_cfg, prov, device)
    model = create_model(config_builder.build_model_config(cfg), env, device)
    env.setup(model.get_reward_model(), model.get_reward_multipliers())
    rl = model.rl_model
    rl.learn(total_timesteps=warmup, progress_bar=False, log_interval=10 ** 9, reset_num_timesteps=True)
    if device == "mps":
        torch.mps.synchronize()
    started = time.time()
    rl.learn(total_timesteps=budget, progress_bar=False, log_interval=10 ** 9, reset_num_timesteps=False)
    if device == "mps":
        torch.mps.synchronize()
    return time.time() - started


class _BenchTimeout(Exception):
    pass


def _time_device_bounded(model_name, device, budget, warmup, prov, env_cfg, timeout_s):
    """`_time_device` with a best-effort per-device wall-clock cap (SIGALRM). A device that stalls (e.g. an
    MPS op that falls back / hangs) is raised as an error and caught per-device, so the OTHER device still
    reports and a summary is always written — instead of the whole probe blowing the compute-runner timeout
    and persisting nothing (which surfaces as 'Benchmark did not settle')."""
    if not (timeout_s and timeout_s > 0 and hasattr(signal, "SIGALRM")):
        return _time_device(model_name, device, budget, warmup, prov, env_cfg)

    def _on_alarm(signum, frame):
        raise _BenchTimeout(f"exceeded {timeout_s}s cap")

    prev = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(int(timeout_s))
    try:
        return _time_device(model_name, device, budget, warmup, prov, env_cfg)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="trainer.bench_device")
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--model-name", default=os.environ.get("BENCH_MODEL_NAME", "reppo-custom"))
    args = parser.parse_args(argv)
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # A device-SPEED probe only needs enough post-warmup steps for a stable us/step (warmup covers the
    # learning_starts=200 buffer-fill + graph compile, so the timed budget is real training). Kept small so
    # the whole cpu+mps run finishes well inside the compute-runner + viewer 10-min windows; tune via env.
    budget = int(os.environ.get("BENCH_BUDGET", "300"))
    warmup = int(os.environ.get("BENCH_WARMUP", "200"))
    device_timeout = int(os.environ.get("BENCH_DEVICE_TIMEOUT", "240"))
    window = os.environ.get("BENCH_WINDOW", "2024")
    model_name = args.model_name

    from trainer import config_builder
    from src.data.data_factory import create_provider

    data_cfg = config_builder.build_data_config({"asset": "BTCUSDT", "timeframe": "1h",
                                                 "walk_forward_window": window,
                                                 "data_type": "only_price_percent", "use_indicators": True})
    env_cfg = config_builder.build_env_config({"timeframe": "1h"})
    prov = create_provider(data_cfg, data_cfg.test_data_paths, data_cfg.fidelity_input_test,
                           data_cfg.fidelity_run_test, data_cfg.layers_test,
                           data_cfg.buyreward_maxwait_test, data_cfg.buyreward_percent_test)

    try:
        import torch

        if "mps" in _available_devices():
            a = torch.randn(1024, 1024, device="mps")
            for _ in range(5):
                a = (a @ a).tanh()
            torch.mps.synchronize()
    except Exception:
        pass

    # cpu is always probed first (it can't be skipped + sets the baseline). Each LATER device gets an
    # ADAPTIVE cap = a multiple of the fastest time already seen (with a floor) — so a device that can't keep
    # within ~SLOWNESS_FACTOR x the best is declared slower in seconds instead of burning the full backstop.
    # This is the common case on Apple silicon, where MPS' per-kernel launch overhead makes these small RL
    # nets far slower than cpu; a model where MPS genuinely WINS still finishes well inside the cap.
    slowness_factor = float(os.environ.get("BENCH_SLOWNESS_FACTOR", "4"))
    min_device_seconds = int(os.environ.get("BENCH_MIN_DEVICE_SECONDS", "30"))
    timings, errors = {}, {}
    best_so_far = None
    for device in _available_devices():
        cap = device_timeout
        if best_so_far is not None:
            cap = min(cap, max(min_device_seconds, int(slowness_factor * best_so_far)))
        try:
            secs = _time_device_bounded(model_name, device, budget, warmup, prov, env_cfg, cap)
            timings[device] = secs
            best_so_far = secs if best_so_far is None else min(best_so_far, secs)
            print(f"{model_name} on {device}: {secs:.2f}s for {budget} steps", flush=True)
        except _BenchTimeout as exc:
            errors[device] = f"too slow — {exc}"
            print(f"{model_name} on {device}: SKIPPED ({errors[device]})", flush=True)
        except Exception as exc:
            errors[device] = f"{type(exc).__name__}: {str(exc)[:160]}"
            print(f"{model_name} on {device}: FAILED {errors[device]}", flush=True)

    out = build_device_summary(model_name, timings, budget, errors or None)
    with open(args.summary_out, "w") as f:
        json.dump(out, f)
    db = out["deviceBenchmark"]
    print(f"best device for {model_name}: {db['bestDevice']} (speedup {db['speedup']}x) -> {args.summary_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
