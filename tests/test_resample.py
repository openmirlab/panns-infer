"""Original recorded and live-reference resampling prove the verbatim closure.

Synthetic CC0 fixtures predate extraction; exact digests require matching builds.
Reads: captured reference outputs and optional unchanged TorchAudio 2.7.1.
"""

import hashlib
import importlib
import inspect
import json
import platform
from pathlib import Path

import numpy as np
import pytest
import torch

from panns_infer._impl import resample as extracted


def test_original_resampling_fixture(device):
    root = Path(__file__).parent / "fixtures" / ("baseline_resample_" + device.split(":")[0])
    metadata = json.loads((root / "environment.json").read_text())
    artifact = root / "resample.npz"
    assert metadata["license"] == "CC0-1.0"
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == metadata["fixture_sha256"]
    if torch.__version__ != metadata["torch"] or platform.machine() != metadata["machine"]:
        pytest.skip("Original float digests require the recording Torch build and architecture")
    if device.startswith("cuda") and torch.cuda.get_device_name(0) != metadata["hardware"]:
        pytest.skip("Original CUDA float digests require the recording GPU")
    if device == "cpu" and torch.backends.cpu.get_cpu_capability() != metadata["cpu_capability"]:
        pytest.skip("Original CPU float digests require the recording CPU dispatch capability")
    with np.load(artifact) as arrays:
        for case in metadata["cases"]:
            audio = torch.from_numpy(arrays[case["id"] + "_input"]).to(device)
            actual = extracted.resample(audio, case["sample_rate"], case["target_rate"])
            np.testing.assert_array_equal(actual.cpu().numpy(), arrays[case["id"] + "_output"])


def reference_module():
    try:
        reference = importlib.import_module("torchaudio")
        if reference.__version__.split("+")[0] != "2.7.1":
            pytest.skip("Live extraction reference is pinned to TorchAudio 2.7.1")
        return importlib.import_module("torchaudio.functional.functional")
    except (ImportError, OSError) as error:
        pytest.skip(f"Optional TorchAudio original reference unavailable: {error}")


@pytest.mark.parametrize("exception", [ImportError, OSError])
def test_unavailable_binary_skips_only_optional_reference(exception, monkeypatch):
    original_import = importlib.import_module

    def failing_import(name, *args, **kwargs):
        if name == "torchaudio":
            raise exception("simulated incompatible binary library")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", failing_import)
    with pytest.raises(
        pytest.skip.Exception, match="Optional TorchAudio original reference unavailable"
    ):
        reference_module()

    # The mandatory package audio boundary remains executable under the same failure.
    from panns_infer._impl.audio import normalize_audio

    waveform, metadata = normalize_audio(torch.zeros(9920), 32000, "cpu")
    assert waveform.shape == (1, 9920)
    assert metadata.model_samples == 9920


def test_reference_source_is_verbatim():
    original = reference_module()
    for name in ["resample", "_get_sinc_resample_kernel", "_apply_sinc_resample_kernel"]:
        assert inspect.getsource(getattr(extracted, name)) == inspect.getsource(
            getattr(original, name)
        )


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_live_original_full_resampling_ab(rate, dtype, device):
    original = reference_module()
    generator = torch.Generator().manual_seed(20261001)
    audio = torch.rand((2, 731), generator=generator, dtype=dtype).to(device)
    expected = original.resample(audio, rate, 32000)
    actual = extracted.resample(audio, rate, 32000)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_passthrough_identity():
    audio = torch.zeros(128)
    assert extracted.resample(audio, 32000, 32000) is audio
