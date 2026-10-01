"""Capture CC0 synthetic resampling fixtures from unchanged TorchAudio 2.7.1.

Run before extraction; this reference-only tool requires optional TorchAudio.
Reads: pinned reference function, seeded signals and explicitly selected device.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
import torchaudio

SOURCE_REVISION = "95c61b4168fc5133be8dd8c1337d929d066ae6cf"
SOURCE_SHA256 = "73caabde698f5cb8b8d0de0d08b7291ef41e508ba836d6ecce4b23b6bb82d798"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    args = parser.parse_args()
    assert torchaudio.__version__.split("+")[0] == "2.7.1"
    source = Path(torchaudio.functional.resample.__code__.co_filename)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(20261001)
    arrays, cases = {}, []
    for rate in [8000, 16000, 22050, 44100, 48000]:
        for samples in [17, (9920 * rate + 31999) // 32000 - 1]:
            for channels in [1, 2]:
                case = f"r{rate}_n{samples}_c{channels}"
                shape = (samples,) if channels == 1 else (channels, samples)
                audio = rng.standard_normal(shape, dtype=np.float32) * np.float32(0.1)
                tensor = torch.from_numpy(audio).to(args.device)
                first = torchaudio.functional.resample(tensor, rate, 32000)
                second = torchaudio.functional.resample(tensor, rate, 32000)
                assert torch.equal(first, second)
                arrays[case + "_input"] = audio
                arrays[case + "_output"] = first.cpu().numpy()
                cases.append({"id": case, "sample_rate": rate, "target_rate": 32000})
    root = (
        Path(__file__).resolve().parents[1]
        / "tests/fixtures"
        / ("baseline_resample_" + args.device.split(":")[0])
    )
    root.mkdir(parents=True, exist_ok=True)
    artifact = root / "resample.npz"
    np.savez_compressed(artifact, **arrays)
    metadata = {
        "license": "CC0-1.0",
        "seed": 20261001,
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "fixture_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "device": args.device,
        "machine": platform.machine(),
        "threads": 4,
        "hardware": torch.cuda.get_device_name(0) if args.device != "cpu" else platform.processor(),
        "tf32": False,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "cases": cases,
    }
    (root / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Captured {len(cases)} original resampling cases on {args.device}")


if __name__ == "__main__":
    main()
