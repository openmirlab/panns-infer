"""Record original PANNs CNN14 evaluation outputs before extracting its runtime.

The upstream tree is only imported as a read-only reference. Fixtures contain
only generated CC0 audio and model activations, never model weights or local paths.
Reads: upstream models.py, official checkpoint, generated fixture manifest.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

p = argparse.ArgumentParser()
p.add_argument("--upstream", type=Path, required=True)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--device", default="cpu")
a = p.parse_args()
sys.path.insert(0, str(a.upstream / "pytorch"))
from models import Cnn14  # noqa: E402 -- read-only upstream import after explicit path

root = Path(__file__).resolve().parents[1] / "tests/fixtures"
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model = Cnn14(32000, 1024, 320, 64, 50, 14000, 527).eval().to(a.device)
model.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=True)["model"])
out = root / ("baseline_" + a.device.split(":")[0])
out.mkdir(exist_ok=True)
for fixture in json.loads((root / "manifest.json").read_text()):
    x, sr = sf.read(root / fixture["path"], dtype="float32", always_2d=True)
    wave = torch.from_numpy(x.T.copy()).mean(0)
    if sr != 32000:
        wave = torchaudio.functional.resample(wave, sr, 32000)
    wave = wave[None].to(a.device)
    with torch.inference_mode():
        spec = model.spectrogram_extractor(wave)
        mel = model.logmel_extractor(spec)
        first = model(wave)
        second = model(wave)
    assert all(torch.equal(first[k], second[k]) for k in first)
    np.savez_compressed(
        out / (fixture["id"] + ".npz"),
        waveform=wave.cpu().numpy(),
        logmel=mel.cpu().numpy(),
        **{k: v.cpu().numpy() for k, v in first.items()},
    )
metadata = {
    "upstream": "qiuqiangkong/audioset_tagging_cnn",
    "source_revision": "d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4",
    "torch": torch.__version__,
    "torchaudio": torchaudio.__version__,
    "numpy": np.__version__,
    "device": a.device,
    "checkpoint_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
    "normalization": "soundfile float32; channel mean before torchaudio functional resample to32000; upstream model exact evaluation path; historic librosa file resampling not claimed",
}
(out / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata))
