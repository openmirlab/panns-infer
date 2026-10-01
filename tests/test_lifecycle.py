"""Session contracts prove reuse, state transitions and real model release.

Offline lifecycle failures need no weights; real ownership is checkpoint-gated.
Reads: public session and optional verified local checkpoint.
"""

import json
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
        assert not tmp_path.exists() or not list(tmp_path.iterdir())
    assert s.status == "closed"
    s.close()
    for operation in [s.load, s.release, lambda: s.infer("x"), s.__enter__]:
        with pytest.raises(RuntimeError, match="closed"):
            operation()


def test_failed_load_retry(tmp_path):
    s = PANNsSession(device="cpu", cache_dir=tmp_path, offline=True)
    for _ in range(2):
        with pytest.raises(FileNotFoundError):
            s.load()
        assert s.status == "failed"
    s.release()
    assert s.status == "released"
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
    reference = weakref.ref(s._model)
    result = s.infer(path)
    assert s.infer(path) == result
    assert len(count) == 1
    with pytest.raises(ValueError):
        s.infer(path, sample_rate=16000)
    assert s.status == "ready"
    s.release()
    assert reference() is None
    assert len(json.loads(json.dumps(asdict(result)))["embedding"]) == 2048
    with pytest.raises(RuntimeError):
        s.infer(path)
    s.load()
    assert len(count) == 2
    assert s.infer(path) == result
    s.close()
    assert s.status == "closed"
