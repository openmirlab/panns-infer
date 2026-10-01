"""Packaging boundaries stay lazy, pinned and free of research runtime imports.

Public CC0 fixture hashes are checked before their expected outputs are trusted.
Reads: package resources, fixture manifests and isolated import subprocesses.
"""

import hashlib
import importlib.resources
import json
import os
import subprocess
import sys
from pathlib import Path

from panns_infer import __version__


def test_lazy_import():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys; from panns_infer import PANNsSession; s=PANNsSession(offline=True); s.close(); assert "torch" not in sys.modules',
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")},
    )
    assert result.returncode == 0, result.stderr


def test_version_and_registry():
    assert __version__ == "0.1.0"
    registry = json.loads(
        importlib.resources.files("panns_infer").joinpath("config/checkpoints.json").read_text()
    )
    assert registry["model_license"] == "CC-BY-4.0"
    assert registry["files"]["Cnn14.pth"]["size"] == 327428481


def test_fixture_hashes():
    root = Path(__file__).parent / "fixtures"
    for f in json.loads((root / "manifest.json").read_text()):
        assert f["license"] == "CC0-1.0"
        assert hashlib.sha256((root / f["path"]).read_bytes()).hexdigest() == f["sha256"]
