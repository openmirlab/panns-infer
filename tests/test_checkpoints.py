"""Verify package-owned cache semantics, checksums and download cleanup offline.

Small synthetic assets exercise integrity rather than duplicating model weights.
Reads: injected artifact registry, temporary paths and mocked HTTPS responses.
"""

import hashlib
import io
import json

import pytest

from panns_infer import checkpoints as c


@pytest.fixture
def registry():
    registry = c._load_registry()
    for meta in registry["files"].values():
        meta["size"] = 4
        meta["sha256"] = hashlib.sha256(b"data").hexdigest()
    return registry


@pytest.fixture
def store(tmp_path, registry, monkeypatch):
    monkeypatch.setattr(c, "_load_registry", lambda: registry)
    return c.CheckpointStore(cache_dir=tmp_path / "cache", offline=True)


def materialize(store, payload=b"data"):
    store.cache_dir.mkdir(parents=True)
    for name in store._registry["files"]:
        (store.cache_dir / name).write_bytes(payload)


def test_inspection_readonly(store):
    first = store.cache_info()
    assert not store.cache_dir.exists()
    assert all(x["integrity"] == "not_checked" for x in first["files"].values())
    with pytest.raises(FileNotFoundError):
        store.resolve()
    assert not store.cache_dir.exists()


def test_verified_offline(store):
    materialize(store)
    assert len(store.resolve()) == 2


@pytest.mark.parametrize("payload", [b"x", b"fail"])
def test_corruption_rejected(store, payload):
    materialize(store, payload)
    with pytest.raises(ValueError):
        store.resolve()


@pytest.mark.parametrize(
    "key,value", [("revision", "../escape"), ("source_revision", "bad"), ("updated_at", "bad")]
)
def test_registry_identity(registry, key, value):
    registry[key] = value
    with pytest.raises(ValueError):
        c._validate_registry(registry)


@pytest.mark.parametrize(
    "key,value", [("sha256", "bad"), ("size", True), ("url", "http://example.com/x")]
)
def test_registry_artifact(registry, key, value):
    registry["files"]["Cnn14.pth"][key] = value
    with pytest.raises(ValueError):
        c._validate_registry(registry)


class Response(io.BytesIO):
    def geturl(self):
        return "https://example.com/asset"


def test_atomic_download(store, monkeypatch):
    store.offline = False
    monkeypatch.setattr(c, "urlopen", lambda *a, **k: Response(b"data"))
    files = store.resolve()
    assert all(x.read_bytes() == b"data" for x in files.values())
    assert not list(store.cache_dir.glob("*.tmp"))


def test_failed_download_cleanup(store, monkeypatch):
    store.offline = False
    monkeypatch.setattr(c, "urlopen", lambda *a, **k: Response(b"fail"))
    with pytest.raises(ValueError):
        store.resolve()
    assert not list(store.cache_dir.glob("*.tmp"))
    assert not (store.cache_dir / "Cnn14.pth").exists()


def test_explicit_missing_never_downloads(tmp_path, registry, monkeypatch):
    monkeypatch.setattr(c, "_load_registry", lambda: registry)
    monkeypatch.setattr(
        c, "urlopen", lambda *a, **k: pytest.fail("explicit paths must not download")
    )
    store = c.CheckpointStore(checkpoint_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        store.resolve()


def test_generic_config_override(tmp_path, registry):
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(registry))
    store = c.CheckpointStore(checkpoint_config=path, cache_dir=tmp_path / "cache")
    assert (
        store.provenance()["files"]["Cnn14.pth"]["sha256"]
        == registry["files"]["Cnn14.pth"]["sha256"]
    )
