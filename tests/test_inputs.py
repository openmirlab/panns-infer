"""Input contracts reject invalid data before expensive model execution.

File, array and tensor inputs share metadata and waveform normalization.
Reads: private audio boundary and generated CC0 files.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from panns_infer._impl.audio import normalize_audio

ROOT = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("rate", [None, 0, -1, True, 3.5, "32000"])
def test_invalid_rate(rate):
    with pytest.raises(ValueError, match="sample_rate"):
        normalize_audio(np.zeros(16000, dtype=np.float32), rate, "cpu")


@pytest.mark.parametrize(
    "data",
    [
        np.zeros(0, dtype=np.float32),
        np.zeros((0, 16000), dtype=np.float32),
        np.zeros((2, 2, 16000), dtype=np.float32),
        np.zeros(16000, dtype=np.int16),
        np.full(16000, np.nan),
        np.full(16000, np.inf),
    ],
)
def test_invalid_waveform(data):
    with pytest.raises(ValueError):
        normalize_audio(data, 32000, "cpu")


def test_short_not_padded():
    with pytest.raises(ValueError, match="at least"):
        normalize_audio(torch.zeros(9919), 32000, "cpu")
    wave, _ = normalize_audio(torch.zeros(9920), 32000, "cpu")
    assert wave.shape == (1, 9920)


def test_file_numpy_tensor():
    path = ROOT / "audio/tonal_sequence_44100_2ch.wav"
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    expected, info = normalize_audio(path, None, "cpu")
    for audio in [data.T, torch.from_numpy(data.T.copy())]:
        actual, metadata = normalize_audio(audio, rate, "cpu")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert info == metadata
    with pytest.raises(ValueError, match="metadata"):
        normalize_audio(path, 16000, "cpu")


def test_type_rejected():
    with pytest.raises(TypeError):
        normalize_audio([0.0] * 16000, 32000, "cpu")


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        normalize_audio(ROOT / "missing.wav", None, "cpu")
