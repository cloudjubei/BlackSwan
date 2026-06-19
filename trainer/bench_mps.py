"""Benchmark BlackSwan training on MPS (Apple Metal) vs CPU across hardcoded setups.

Runs the SAME (data, env, model) configuration on each device through the real trainer
path (``trainer.run._run_one``) and reports training wall-time, so you can see where the
GPU helps. The setups span simple -> complex:

    dqn          off-policy, tiny MLP   [64, 64]          (small per-step compute)
    ppo          on-policy,  tiny MLP   [64, 64]
    ppo-wide     on-policy,  large MLP  [2048, 1024, 512] (matmul-heavy -> favours GPU)
    reppo-custom on-policy,  LSTM       (the default production model)

Usage:
    .venv/bin/python -m trainer.bench_mps                         # default setups, cpu + mps
    .venv/bin/python -m trainer.bench_mps --setups dqn,ppo-wide   # pick setups
    .venv/bin/python -m trainer.bench_mps --devices cpu           # cpu only
    .venv/bin/python -m trainer.bench_mps --window 2024 --episodes 2
    BS_NUM_THREADS=8 .venv/bin/python -m trainer.bench_mps        # give CPU more threads

Notes:
  - 1d klines are used (fastest path). ``--window`` picks the walk-forward split.
  - DQN trains for the data length; the on-policy models train a fixed step budget, so their
    times are independent of ``--window``.
  - CPU thread count comes from BS_NUM_THREADS (default 2, matching the Model Trainer's
    parallel-run cap). MPS uses the single GPU regardless. Raise it for a single-run-uses-all
    -cores comparison.
  - MPS shares ONE GPU, so many concurrent MPS runs contend; CPU runs parallelise across cores.
    A per-run speedup here does not imply a faster full sweep.
"""

import argparse
import time

from trainer import config_builder
from trainer import run as trainer_run

try:
    import torch

    _MPS_OK = torch.backends.mps.is_available()
except Exception:
    torch = None
    _MPS_OK = False

_BASE = {
    "reward_model": "combo_all2",
    "learning_rate": 0.0001,
    "gamma": 0.99,
    "batch_size": 64,
    "buffer_size": 5000,
    "learning_starts": 50,
    "episodes": 1,
    "timeframe": "1d",
    "walk_forward_window": "2022",
    "seed": 0,
}

SETUPS = {
    "dqn": {"model_name": "dqn", "net_arch": "64,64"},
    "ppo": {"model_name": "ppo", "net_arch": "64,64"},
    "ppo-wide": {"model_name": "ppo", "net_arch": "2048,1024,512", "batch_size": 512},
    "reppo-custom": {"model_name": "reppo-custom", "net_arch": "64,64"},
}

_DEFAULT_SETUPS = ["dqn", "ppo", "ppo-wide", "reppo-custom"]


def _cfg(setup_name, device, window, episodes):
    cfg = dict(_BASE, **SETUPS[setup_name])
    cfg["device"] = device
    cfg["walk_forward_window"] = window
    cfg["episodes"] = episodes
    return cfg


def _warmup_mps():
    """Pay MPS's one-time context/kernel init once, so it isn't charged to the first timed run."""
    if not _MPS_OK:
        return
    a = torch.randn(512, 512, device="mps")
    for _ in range(3):
        a = (a @ a).tanh()
    torch.mps.synchronize()


def _empty_cache():
    if _MPS_OK and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _run(setup_name, device, window, episodes):
    """Return (train_seconds, objective, n_trades, device_used) or raise."""
    cfg = _cfg(setup_name, device, window, episodes)
    env_test, state, model, is_rl, train_seconds = trainer_run._run_one(cfg)
    device_used = str(getattr(getattr(model, "rl_model", None), "policy", None).device)
    objective = float(state[2]) if state is not None and len(state) > 2 else float("nan")
    n_trades = int(state[17]) if state is not None and len(state) > 17 else -1
    _empty_cache()
    return train_seconds, objective, n_trades, device_used


def main(argv=None):
    parser = argparse.ArgumentParser(prog="trainer.bench_mps")
    parser.add_argument("--setups", default=",".join(_DEFAULT_SETUPS),
                        help=f"comma list from {list(SETUPS)} or 'all'")
    parser.add_argument("--devices", default="cpu,mps", help="comma list: cpu,mps")
    parser.add_argument("--window", default="2022", help="walk-forward window id")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args(argv)

    setups = list(SETUPS) if args.setups == "all" else [s.strip() for s in args.setups.split(",") if s.strip()]
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if "mps" in devices and not _MPS_OK:
        print("MPS not available on this host — dropping it from the comparison.")
        devices = [d for d in devices if d != "mps"]

    if not _data_ok():
        print("1d BTCUSDT klines not on disk — cannot benchmark.")
        return 1

    threads = torch.get_num_threads() if torch is not None else "?"
    print(f"torch={getattr(torch, '__version__', '?')} cpu_threads={threads} mps_available={_MPS_OK}")
    print(f"setups={setups} devices={devices} window={args.window} episodes={args.episodes}\n")

    _warmup_mps()

    rows = []
    for setup_name in setups:
        if setup_name not in SETUPS:
            print(f"!! unknown setup '{setup_name}' — skipping"); continue
        times = {}
        for device in devices:
            label = f"{setup_name} on {device}"
            try:
                train_s, obj, trades, dev_used = _run(setup_name, device, args.window, args.episodes)
                times[device] = train_s
                print(f"  {label:32s} train={train_s:7.2f}s  device={dev_used:7s} return={obj*100:+6.2f}% trades={trades}")
            except Exception as exc:
                print(f"  {label:32s} FAILED: {exc!r}")
        rows.append((setup_name, times))

    print("\n" + "=" * 64)
    header = f"{'setup':16s}" + "".join(f"{d+' (s)':>12s}" for d in devices)
    if "cpu" in devices and "mps" in devices:
        header += f"{'cpu/mps':>10s}"
    print(header)
    print("-" * 64)
    for setup_name, times in rows:
        line = f"{setup_name:16s}" + "".join(f"{times.get(d, float('nan')):>12.2f}" for d in devices)
        if "cpu" in devices and "mps" in devices and times.get("mps"):
            line += f"{times['cpu'] / times['mps']:>10.2f}x"
        print(line)
    print("=" * 64)
    print("cpu/mps > 1.0 => MPS trained faster for that setup.")
    return 0


def _data_ok():
    try:
        config_builder.require_data_present(dict(_BASE))
        return True
    except SystemExit:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
