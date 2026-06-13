"""BlackSwan's dip/regression line: a binary "will the next buy be profitable?"
classifier (RegressionPredictEnv + the tuned `model_regression` MLP), judged on f1.

A separate trainer-conformant line from the trading manifest — one objective per
project: the objective is f1 (not Sharpe), and the env / model / summary differ.
Runs on the same on-disk daily klines via the fast single-layer provider, so a dip
campaign is quick to verify without the heavy sub-daily multi-timeline data.
"""

import copy
import math

from omegaconf import OmegaConf

from src.conf.data_config import DataConfig
from src.conf.env_config import EnvConfig
from src.conf.model_config import model_regression_dip
from src.model.model_factory import get_model_combinations

from trainer import config_builder


def require_data_present(cfg=None):
    config_builder.require_data_present(cfg)


def build_data_config(cfg):
    asset = str(cfg.get("asset", "BTCUSDT"))
    maxwait = int(cfg.get("buyreward_maxwait", 5))
    percent = float(cfg.get("buyreward_percent", 0.02))
    return OmegaConf.structured(
        DataConfig(
            id=f"{asset}-1d-dip",
            train_data_paths=[config_builder._daily_files(config_builder._TRAIN_PAIRS, asset)],
            test_data_paths=[config_builder._daily_files(config_builder._TEST_PAIRS, asset)],
            lookback_window_size=1,
            type=str(cfg.get("data_type", "only_price_percent")),
            timestamp="none",
            buyreward_percent=percent,
            buyreward_maxwait=maxwait,
            fidelity_input="1d",
            fidelity_run="1d",
            layers=["1d"],
            fidelity_input_test="1d",
            fidelity_run_test="1d",
            layers_test=["1d"],
            buyreward_percent_test=percent,
            buyreward_maxwait_test=maxwait,
        )
    )


def build_env_config(cfg):
    return OmegaConf.structured(
        EnvConfig(
            type="regression_predict",
            observations_contain=[],
            batch_size=int(cfg.get("batch_size", 32)),
        )
    )


def build_model_config(cfg):
    """Return one concrete regression ModelConfig for the lever values in ``cfg``."""
    search = copy.deepcopy(model_regression_dip)
    reg = search.model_regression
    reg.learning_rate = [float(cfg.get("learning_rate", 0.0001))]
    reg.loss_fn = [str(cfg.get("loss_fn", "bcelogits"))]
    reg.episodes = [int(cfg.get("episodes", 1))]
    reg.seed = int(cfg["seed"]) if cfg.get("seed") is not None else None
    reg.pos_weight = float(cfg.get("pos_weight", 0.0))
    reg.decision_threshold = float(cfg.get("decision_threshold", 0.5))
    if cfg.get("checkpoint_to_load"):
        reg.checkpoint_to_load = str(cfg["checkpoint_to_load"])
    if "net_arch" in cfg:
        reg.net_arch = [config_builder._parse_net_arch(cfg["net_arch"])]
    if "activation_fn" in cfg:
        reg.activation_fn = [str(cfg["activation_fn"])]
    config = get_model_combinations(OmegaConf.structured(search))[0]
    config.iterations_to_pick_best = 1
    return config


def _finite(x, default=0.0):
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else default


def _health(state):
    precision = _finite(state[3]) if len(state) > 3 else 0.0
    recall = _finite(state[4]) if len(state) > 4 else 0.0
    flags = []
    if precision + recall == 0:
        flags.append("no_positive_signal")
    return {"status": "degenerate" if flags else "ok", "flags": flags}


def build_summary(env, state, cfg, model, ran_at):
    """The trainer-standard RunSummary for one dip/regression test run (f1 objective)."""
    f1 = _finite(state[0]) if len(state) > 0 else 0.0
    n_pos = int(getattr(env, "n_positive", 0))
    n_neg = int(getattr(env, "n_negative", 0))
    total = n_pos + n_neg
    summary = {
        "objective": f1,
        "metrics": {
            "f1": f1,
            "simple_ratio": _finite(state[1]) if len(state) > 1 else 0.0,
            "accuracy": _finite(state[2]) if len(state) > 2 else 0.0,
            "precision": _finite(state[3]) if len(state) > 3 else 0.0,
            "recall": _finite(state[4]) if len(state) > 4 else 0.0,
            "negative_recall": _finite(state[5]) if len(state) > 5 else 0.0,
            "positive_rate": (n_pos / total) if total else 0.0,
        },
        "health": _health(state),
        "config": dict(cfg),
        "provenance": {"ranAt": ran_at},
        "dataset": {
            "asset": str(cfg.get("asset", "BTCUSDT")),
            "timeframe": "1d",
            "candles": len(getattr(env, "predictions", []) or []),
        },
    }
    checkpoint = getattr(model, "id", None)
    if checkpoint:
        summary["artifacts"] = {"checkpoint": f"checkpoints/{checkpoint}", "best": False}
    if "seed" in cfg:
        summary["seed"] = int(cfg["seed"])
        summary["provenance"]["seed"] = int(cfg["seed"])
    return summary
