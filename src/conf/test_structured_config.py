import pytest
from hydra.core.config_store import ConfigStore

from src.conf import structured_config as sc
from src.conf.structured_config import Config, Repo


# ---------------------------------------------------------------------------
# Repo dataclass : plain string defaults
# ---------------------------------------------------------------------------


def test_repo_defaults():
    r = Repo()
    assert r.repo_owner == "cloudjubei"
    assert r.repo_name == "black-swan"


def test_repo_override():
    r = Repo(repo_owner="someone", repo_name="other")
    assert r.repo_owner == "someone"
    assert r.repo_name == "other"


# ---------------------------------------------------------------------------
# Config dataclass : scalar defaults + nested Repo factory
# ---------------------------------------------------------------------------


def test_config_scalar_defaults_when_lists_supplied():
    # The list fields have a broken default_factory (see xfail below); supply them explicitly to
    # exercise the scalar defaults.
    c = Config(data_configs=[], env_configs=[], model_configs=[])
    assert c.user == "cloudjubei"
    assert c.experiment_name == "hiperparams"
    assert c.run_name == "test"
    assert c.device == "cpu"
    assert c.local_only is True
    assert c.show_render is True


def test_config_dagshub_repo_is_fresh_repo_instance():
    # default_factory=Repo builds a fresh Repo per Config; distinct instances must not be shared.
    a = Config(data_configs=[], env_configs=[], model_configs=[])
    b = Config(data_configs=[], env_configs=[], model_configs=[])
    assert isinstance(a.dagshub_repo, Repo)
    assert a.dagshub_repo is not b.dagshub_repo
    assert a.dagshub_repo.repo_owner == "cloudjubei"


def test_config_accepts_overrides():
    c = Config(
        run_name="myrun",
        device="mps",
        local_only=False,
        data_configs=[1],
        env_configs=[2],
        model_configs=[3],
    )
    assert c.run_name == "myrun"
    assert c.device == "mps"
    assert c.local_only is False
    assert c.data_configs == [1]


@pytest.mark.xfail(
    reason="BUG: data_configs/env_configs/model_configs use field(default_factory=List) where "
    "List is typing.List (a generic alias), not the list constructor. Constructing Config() with "
    "no list args raises 'Type List cannot be instantiated; use list() instead'. Should be "
    "default_factory=list.",
    strict=False,
    raises=TypeError,
)
def test_config_constructs_with_default_lists():
    # The intended contract: Config() yields empty lists for the *_configs fields.
    c = Config()
    assert c.data_configs == []
    assert c.env_configs == []
    assert c.model_configs == []


# ---------------------------------------------------------------------------
# Module import side-effect : the ConfigStore registration at module load.
# ---------------------------------------------------------------------------


def test_module_registers_config_simple_node():
    # Importing structured_config runs `cs.store(name="config_simple", node=Config(...))`.
    cs = ConfigStore.instance()
    assert "config_simple.yaml" in cs.repo


def test_registered_node_has_expected_run_name():
    cs = ConfigStore.instance()
    node = cs.repo["config_simple.yaml"].node
    # stored via Config(run_name="test-simple", ...)
    assert node.run_name == "test-simple"


def test_module_cs_is_configstore_singleton():
    # The module-level `cs` is the process-wide ConfigStore singleton.
    assert sc.cs is ConfigStore.instance()
