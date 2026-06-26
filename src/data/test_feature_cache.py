"""Direct tests for the feature-frame disk cache (feature_cache.py)."""

import os

from src.data import feature_cache


def _touch(path, content="x"):
    with open(path, "w") as f:
        f.write(content)


def test_load_or_build_builds_on_miss_then_hits_cache(tmp_path):
    src = tmp_path / "a.json"
    _touch(str(src))
    cdir = str(tmp_path / "cache")
    calls = []

    def build():
        calls.append(1)
        return {"v": 42}

    r1 = feature_cache.load_or_build([str(src)], {"p": 1}, build, cache_dir=cdir)
    r2 = feature_cache.load_or_build([str(src)], {"p": 1}, build, cache_dir=cdir)
    assert r1 == r2 == {"v": 42}
    assert len(calls) == 1  # the second call hit the cache


def test_source_content_change_invalidates(tmp_path):
    src = tmp_path / "a.json"
    _touch(str(src), "one")
    cdir = str(tmp_path / "cache")
    feature_cache.load_or_build([str(src)], {"p": 1}, lambda: "first", cache_dir=cdir)
    _touch(str(src), "a much longer content string")  # size change -> new signature
    r = feature_cache.load_or_build([str(src)], {"p": 1}, lambda: "second", cache_dir=cdir)
    assert r == "second"


def test_param_change_invalidates(tmp_path):
    src = tmp_path / "a.json"
    _touch(str(src))
    cdir = str(tmp_path / "cache")
    feature_cache.load_or_build([str(src)], {"p": 1}, lambda: "a", cache_dir=cdir)
    r = feature_cache.load_or_build([str(src)], {"p": 2}, lambda: "b", cache_dir=cdir)
    assert r == "b"


def test_disabled_bypasses_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("BS_FEATURE_CACHE", "0")
    src = tmp_path / "a.json"
    _touch(str(src))
    cdir = str(tmp_path / "cache")
    calls = []

    def build():
        calls.append(1)
        return "x"

    feature_cache.load_or_build([str(src)], {}, build, cache_dir=cdir)
    feature_cache.load_or_build([str(src)], {}, build, cache_dir=cdir)
    assert len(calls) == 2
    assert not os.path.isdir(cdir)


def test_corrupt_entry_rebuilds(tmp_path):
    src = tmp_path / "a.json"
    _touch(str(src))
    cdir = str(tmp_path / "cache")
    os.makedirs(cdir)
    key = feature_cache.compute_key([str(src)], {"p": 1})
    with open(os.path.join(cdir, key + ".pkl"), "wb") as f:
        f.write(b"not a pickle")
    r = feature_cache.load_or_build([str(src)], {"p": 1}, lambda: "rebuilt", cache_dir=cdir)
    assert r == "rebuilt"


def test_nested_path_lists_hash_like_flattened(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _touch(str(a))
    _touch(str(b))
    assert feature_cache.compute_key([[str(a), str(b)]], {}) == feature_cache.compute_key([str(a), str(b)], {})


def test_missing_source_does_not_alias_present(tmp_path):
    present = tmp_path / "a.json"
    _touch(str(present))
    missing = str(tmp_path / "gone.json")
    assert feature_cache.compute_key([str(present)], {}) != feature_cache.compute_key([missing], {})
