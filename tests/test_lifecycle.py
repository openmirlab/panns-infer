"""Session contracts prove reuse, state transitions and real model release.

Offline lifecycle failures need no weights; real ownership is checkpoint-gated.
Reads: public session and optional verified local checkpoint.
"""

import json
import threading
import weakref
from dataclasses import asdict
from pathlib import Path

import pytest

from panns_infer import PANNsSession
from panns_infer.session import _resolve_device


def test_not_implicitly_loaded(tmp_path):
    with PANNsSession(cache_dir=tmp_path, offline=True) as s:
        assert s.status == "new"
        with pytest.raises(RuntimeError, match="load"):
            s.infer("absent.wav")
        with pytest.raises(RuntimeError, match="load"):
            s.loaded_checkpoint_info()
        assert not tmp_path.exists() or not list(tmp_path.iterdir())
    assert s.status == "closed"
    s.close()
    for operation in [
        s.load,
        s.release,
        lambda: s.infer("x"),
        s.__enter__,
        s.loaded_checkpoint_info,
    ]:
        with pytest.raises(RuntimeError, match="closed"):
            operation()


def test_failed_load_retry(tmp_path):
    s = PANNsSession(device="cpu", cache_dir=tmp_path, offline=True)
    for _ in range(2):
        with pytest.raises(FileNotFoundError):
            s.load()
        assert s.status == "failed"
        with pytest.raises(RuntimeError, match="load"):
            s.loaded_checkpoint_info()
    s.release()
    assert s.status == "released"
    with pytest.raises(RuntimeError, match="load"):
        s.loaded_checkpoint_info()
    s.close()


@pytest.mark.parametrize("device", ["bad", "cuda:-1", "cuda:0.0", None, 1])
def test_bad_device(device):
    with pytest.raises(ValueError):
        _resolve_device(device)


def test_unavailable_cuda():
    with pytest.raises(RuntimeError):
        _resolve_device("cuda:999999")


@pytest.mark.model
def test_reuse_release_reload(assets, device, monkeypatch):
    import panns_infer._impl.model as core

    original = core.Cnn14
    count = []

    def factory():
        count.append(1)
        return original()

    monkeypatch.setattr(core, "Cnn14", factory)
    path = Path(__file__).parent / "fixtures/audio/silence_32000_1ch.wav"
    s = PANNsSession(device=device, checkpoint_dir=assets, offline=True)
    s.load()
    s.load()
    loaded = s.loaded_checkpoint_info()
    assert loaded["model_id"] == "panns-cnn14"
    assert loaded["device"] == str(s.device)
    assert loaded["checkpoint_id"] == f"{loaded['repo']}@{loaded['revision']}"
    assert loaded["checkpoint_sha256"] == loaded["files"]["Cnn14.pth"]["sha256"]
    assert all("path" not in artifact for artifact in loaded["files"].values())
    reference = weakref.ref(s._model)
    result = s.infer(path)
    assert dict(result.checkpoint.artifacts)["Cnn14.pth"] == loaded["checkpoint_sha256"]
    assert s.infer(path) == result
    assert len(count) == 1
    with pytest.raises(ValueError):
        s.infer(path, sample_rate=16000)
    assert s.status == "ready"
    s.release()
    with pytest.raises(RuntimeError, match="load"):
        s.loaded_checkpoint_info()
    assert reference() is None
    assert len(json.loads(json.dumps(asdict(result)))["embedding"]) == 2048
    with pytest.raises(RuntimeError):
        s.infer(path)
    s.load()
    assert s.loaded_checkpoint_info() == loaded
    assert len(count) == 2
    assert s.infer(path) == result
    s.close()
    assert s.status == "closed"


@pytest.fixture
def snapshot_session(tmp_path, monkeypatch):
    """Use real integrity resolution with a cheap model stub to test ownership offline."""
    import hashlib

    import panns_infer._impl.model as core
    from panns_infer.checkpoints import _load_registry

    registry = _load_registry()
    payloads = {
        "Cnn14.pth": b"synthetic model",
        "class_labels_indices.csv": (
            "index,mid,display_name\n" + "\n".join(f"{i},/m/test{i},label{i}" for i in range(527))
        ).encode(),
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)
        registry["files"][name].update(
            size=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        )
    config = tmp_path / "registry.json"
    config.write_text(json.dumps(registry))

    class Model:
        def load_checkpoint(self, path):
            assert path.read_bytes() == payloads["Cnn14.pth"]

        def to(self, device):
            return self

        def eval(self):
            return self

        def requires_grad_(self, value):
            return self

    monkeypatch.setattr(core, "Cnn14", Model)
    with PANNsSession(device="cpu", checkpoint_dir=tmp_path, checkpoint_config=config) as session:
        session.load()
        yield session, config


def test_loaded_snapshot_defensive_and_independent(snapshot_session, monkeypatch):
    session, config = snapshot_session
    expected = session.loaded_checkpoint_info()
    assert json.loads(json.dumps(expected)) == expected
    session._store._registry["revision"] = "mutated"
    session._store._registry["files"]["Cnn14.pth"]["sha256"] = "0" * 64
    config.write_text("not JSON anymore")
    for name in expected["files"]:
        (config.parent / name).unlink()
    returned = session.loaded_checkpoint_info()
    returned["files"]["Cnn14.pth"]["sha256"] = "caller mutation"
    returned["audioset_licenses"]["dataset"] = "caller mutation"

    def forbidden(*args, **kwargs):
        raise AssertionError("loaded provenance must not consult files or the store")

    monkeypatch.setattr(session._store, "provenance", forbidden)
    monkeypatch.setattr(session._store, "cache_info", forbidden)
    monkeypatch.setattr(session._store, "resolve", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    assert session.loaded_checkpoint_info() == expected


def test_loaded_snapshot_serializes_with_session_operations(snapshot_session):
    session, _ = snapshot_session
    started, completed = threading.Event(), threading.Event()
    results = []

    def read():
        started.set()
        results.append(session.loaded_checkpoint_info())
        completed.set()

    with session._lock:
        worker = threading.Thread(target=read)
        worker.start()
        assert started.wait(2)
        assert not completed.wait(0.05)
    worker.join(2)
    assert completed.is_set()
    assert results == [session.loaded_checkpoint_info()]


@pytest.mark.parametrize("pad_short", [None, 0, 1, "true"])
def test_session_rejects_padding_option_before_normalization(
    snapshot_session, pad_short, monkeypatch
):
    session, _ = snapshot_session
    import panns_infer._impl.audio as audio

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid padding option must fail before normalization")

    monkeypatch.setattr(audio, "normalize_audio", forbidden)
    with pytest.raises(ValueError, match="pad_short"):
        session.infer("missing.wav", pad_short=pad_short)
    assert session.status == "ready"
