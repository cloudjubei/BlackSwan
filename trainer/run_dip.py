"""Trainer-conformant CLI for BlackSwan's dip/regression line (f1 objective).

  python -m trainer.run_dip --config-json <path> --summary-out <path>
  python -m trainer.run_dip --calibrate --summary-out <path>

Trains the tuned dip MLP (``model_regression``) on the binary "will the next buy be
profitable?" label and writes a thefactory RunSummary scored on f1. Reuses the
unchanged src/ machinery and the trading line's seed/progress/write helpers; never
runs the src.main sweep and never touches Hydra.
"""

import argparse
import datetime
import json
import os
import sys
import time

from src.data.data_factory import create_provider
from src.environment.env_factory import create_environment
from src.model.model_factory import create_model

from trainer import dip
from trainer.run import _progress, _seed_everything, _write

# A deliberately tiny configuration for --calibrate (one short 1d episode).
_CALIBRATE_CFG = {
    "loss_fn": "bcelogits",
    "learning_rate": 0.0001,
    "episodes": 1,
    "batch_size": 32,
    "buyreward_maxwait": 5,
    "buyreward_percent": 0.02,
    "device": "cpu",
}


def _run_one(cfg):
    """Build → train → test the dip classifier; return (env_test, run_state, model, train_seconds)."""
    dip.require_data_present(cfg)
    device = str(cfg.get("device", "cpu"))
    if "seed" in cfg:
        _seed_everything(int(cfg["seed"]))

    _progress("loading")
    data_cfg = dip.build_data_config(cfg)
    env_cfg = dip.build_env_config(cfg)
    model_cfg = dip.build_model_config(cfg)

    provider_train = create_provider(
        data_cfg,
        data_cfg.train_data_paths,
        data_cfg.fidelity_input,
        data_cfg.fidelity_run,
        data_cfg.layers,
        data_cfg.buyreward_maxwait,
        data_cfg.buyreward_percent,
    )
    provider_test = create_provider(
        data_cfg,
        data_cfg.test_data_paths,
        data_cfg.fidelity_input_test,
        data_cfg.fidelity_run_test,
        data_cfg.layers_test,
        data_cfg.buyreward_maxwait_test,
        data_cfg.buyreward_percent_test,
    )
    env_train = create_environment(env_cfg, provider_train, device)
    env_test = create_environment(env_cfg, provider_test, device)
    model = create_model(model_cfg, env_train, device)

    started = time.time()
    if not model.is_pretrained():
        _progress("train", total=int(cfg.get("episodes", 1)))
        model.train(env_train)
    train_seconds = time.time() - started

    _progress("test")
    model.test(env_test, True)
    _progress("summarize")
    return env_test, env_test.get_run_state(), model, train_seconds


def main(argv=None):
    parser = argparse.ArgumentParser(prog="trainer.run_dip")
    parser.add_argument("--config-json")
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--resume-from")
    args = parser.parse_args(argv)

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if args.calibrate:
        cfg = dict(_CALIBRATE_CFG)
    elif args.config_json:
        with open(args.config_json) as f:
            cfg = json.load(f)
    else:
        parser.error("either --config-json or --calibrate is required")

    if args.resume_from:
        cfg["checkpoint_to_load"] = args.resume_from

    ran_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    env_test, state, model, train_seconds = _run_one(cfg)
    out = dip.build_summary(env_test, state, cfg, model, ran_at)

    if args.calibrate:
        episodes = int(cfg.get("episodes", 1))
        out["calibration"] = {
            "unitsPerSecond": (episodes / train_seconds) if train_seconds > 0 else 0.0,
            "secondsObserved": train_seconds,
            "units": episodes,
        }

    _write(args.summary_out, out)
    print(
        f"objective(f1)={out['objective']:.4f} status={out['health']['status']} -> {args.summary_out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
