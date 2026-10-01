"""Exercise an installed wheel through public file inference and full CLI JSON.

Run with python -I after installing the wheel into a fresh environment. Every
embedding/score, offline use and release/reload is checked against original fixtures.
Reads: installed package, CC0 baseline fixtures and verified local asset directory.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import panns_infer
from panns_infer import PANNsSession, checkpoints

p = argparse.ArgumentParser()
p.add_argument("--checkpoint-dir", type=Path, required=True)
p.add_argument("--device", default="cpu")
a = p.parse_args()
assert "site-packages" in str(Path(panns_infer.__file__)), panns_infer.__file__
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
checkpoints.urlopen = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError("offline network use")
)
root = Path(__file__).resolve().parents[1] / "tests/fixtures"
records = {}
with PANNsSession(device=a.device, checkpoint_dir=a.checkpoint_dir, offline=True) as session:
    assert session.status == "new"
    try:
        session.infer(root / "absent.wav")
    except RuntimeError:
        pass
    else:
        raise AssertionError("infer must require ready")
    session.load()
    for f in json.loads((root / "manifest.json").read_text()):
        path = root / f["path"]
        result = session.infer(path)
        gold = np.load(root / ("baseline_" + a.device.split(":")[0]) / (f["id"] + ".npz"))
        np.testing.assert_allclose(result.scores, gold["clipwise_output"][0], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(result.embedding, gold["embedding"][0], rtol=1e-4, atol=1e-4)
        assert len(result.labels) == len(result.label_ids) == 527
        assert result.embedding_shape == (2048,)
        data, rate = sf.read(path, dtype="float32", always_2d=True)
        for audio in [data.T, torch.from_numpy(data.T.copy())]:
            assert session.infer(audio, sample_rate=rate) == result
        records[f["id"]] = result
    path = root / "audio/silence_32000_1ch.wav"
    for audio in [root / "missing.wav", np.array([float("nan")], dtype=np.float32)]:
        try:
            session.infer(audio, sample_rate=32000)
        except (OSError, ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("invalid input accepted")
        assert session.status == "ready"
    session.release()
    assert session.status == "released"
    session.load()
    assert session.infer(path) == records["silence"]
assert session.status == "closed"
assert len(json.loads(json.dumps(asdict(records["silence"])))["embedding"]) == 2048
code = "import torch; torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; from panns_infer.cli import main; main()"
cli = subprocess.run(
    [
        sys.executable,
        "-I",
        "-c",
        code,
        str(path),
        "--device",
        a.device,
        "--checkpoint-dir",
        str(a.checkpoint_dir),
        "--offline",
    ],
    check=True,
    capture_output=True,
    text=True,
)
payload = json.loads(cli.stdout)
assert len(payload["embedding"]) == 2048 and len(payload["scores"]) == 527
np.testing.assert_array_equal(payload["embedding"], records["silence"].embedding)
np.testing.assert_array_equal(payload["scores"], records["silence"].scores)
assert payload["checkpoint"] == json.loads(json.dumps(asdict(records["silence"].checkpoint)))
print(
    json.dumps(
        {
            "status": "passed",
            "device": a.device,
            "installed_package": panns_infer.__file__,
            "fixtures": len(records),
            "cli_scores": 527,
            "cli_embedding_components": 2048,
            "offline": True,
            "release_reload": True,
        }
    )
)
