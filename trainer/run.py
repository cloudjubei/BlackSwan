"""Trainer-conformant CLI for BlackSwan (trading line).

  python -m trainer.run --config-json <path> --summary-out <path>
  python -m trainer.run --calibrate --summary-out <path>

Runs ONE (data, env, model) configuration end-to-end, reusing the unchanged
src/ machinery, and writes a thefactory RunSummary. The Model Trainer sweeps
levers + seeds across many of these; this entry never runs the src.main sweep
and never touches Hydra.
"""

import argparse
import datetime
import json
import os
import random
import sys
import time

import numpy as np

from src.data.data_factory import create_provider
from src.environment.env_factory import create_environment
from src.model.model_factory import create_model

from trainer import config_builder, summary as summary_mod

# A deliberately tiny configuration for --calibrate (one short 1d episode).
_CALIBRATE_CFG = {
    "model_name": "dqn",
    "reward_model": "combo_all2",
    "learning_rate": 0.0001,
    "gamma": 0.99,
    "batch_size": 64,
    "buffer_size": 5000,
    "learning_starts": 50,
    "net_arch": "64,64",
    "episodes": 1,
    "timeframe": "1d",
    "device": "cpu",
}


def _progress(phase, **extra):
    """Emit a structured sub-phase marker the Model Trainer parses for live progress."""
    print("@@PROGRESS " + json.dumps({"phase": phase, **extra}), flush=True)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass
    try:
        from stable_baselines3.common.utils import set_random_seed

        set_random_seed(seed)
    except Exception:
        pass


def _run_one(cfg):
    """Build → train → deterministic test; return (env_test, run_state, model, is_rl, train_seconds)."""
    config_builder.require_data_present(cfg)
    device = str(cfg.get("device", "cpu"))
    if "seed" in cfg:
        _seed_everything(int(cfg["seed"]))

    _progress("loading")
    is_rl = not config_builder.is_hodl(cfg)
    data_cfg = config_builder.build_data_config(cfg)
    env_cfg = config_builder.build_env_config(cfg)
    model_cfg = config_builder.build_model_config(cfg)

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

    env_train.setup(model.get_reward_model(), model.get_reward_multipliers())
    started = time.time()
    if not model.is_pretrained():
        _progress("train", total=env_train.get_timesteps() * int(cfg.get("episodes", 1)))
        model.train(env_train)
    train_seconds = time.time() - started

    env_test.setup(model.get_reward_model(), model.get_reward_multipliers())
    _progress("test", total=env_test.get_timesteps())
    model.test(env_test, True)
    _progress("summarize")
    return env_test, env_test.get_run_state(), model, is_rl, train_seconds


def _write(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="trainer.run")
    parser.add_argument("--config-json")
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--resume-from")
    parser.add_argument("--evaluate", action="store_true")
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

    if args.evaluate:
        # Re-test a saved checkpoint on the test window WITHOUT retraining. The hub injects the run's
        # `checkpoint` artifact into the config; map it to the loader key so the model loads it and
        # `_run_one` skips training (model.is_pretrained()).
        checkpoint = cfg.get("checkpoint") or cfg.get("checkpoint_to_load")
        if not checkpoint:
            parser.error("--evaluate requires a 'checkpoint' (or 'checkpoint_to_load') in the config")
        # artifacts.checkpoint is "<checkpoints_folder>/<id>.zip"; the loader re-adds the folder + .zip,
        # so reduce to the bare id.
        ckpt = os.path.basename(str(checkpoint))
        if ckpt.endswith(".zip"):
            ckpt = ckpt[: -len(".zip")]
        cfg["checkpoint_to_load"] = ckpt

    ran_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    env_test, state, model, is_rl, train_seconds = _run_one(cfg)
    out = summary_mod.build_summary(env_test, state, cfg, model, ran_at, is_rl)

    if args.evaluate:
        out["evaluation"] = {"checkpoint": str(cfg.get("checkpoint_to_load") or ""), "episodes": 0}

    if args.calibrate:
        episodes = int(cfg.get("episodes", 1))
        out["calibration"] = {
            "unitsPerSecond": (episodes / train_seconds) if train_seconds > 0 else 0.0,
            "secondsObserved": train_seconds,
            "units": episodes,
        }

    _write(args.summary_out, out)
    print(
        f"objective(traded_return)={out['objective']:.4f} status={out['health']['status']} -> {args.summary_out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
