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
from panns_infer._impl.resample import resample

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


@pytest.mark.parametrize("samples", [9919, 9920])
def test_padding_boundary(samples):
    source = torch.ones(samples)
    wave, info = normalize_audio(source, 32000, "cpu", pad_short=True)
    assert info.normalized_samples == samples
    assert info.model_samples == 9920
    assert info.padding_samples == 9920 - samples
    assert info.padding_seconds == (9920 - samples) / 32000
    assert info.samples == samples
    assert info.duration_seconds == samples / 32000
    assert torch.equal(wave[0, :samples], source)
    assert torch.count_nonzero(wave[0, samples:]) == 0


@pytest.mark.parametrize("value", [None, 0, 1, "true", np.bool_(True)])
def test_pad_short_strict_bool_before_file_io(value, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid padding option must fail before file IO")

    monkeypatch.setattr(Path, "is_file", forbidden)
    with pytest.raises(ValueError, match="pad_short"):
        normalize_audio(ROOT / "missing.wav", None, "cpu", pad_short=value)


@pytest.mark.parametrize("data", [np.zeros(0), np.array([np.nan]), np.array([np.inf])])
def test_padding_never_accepts_empty_or_nonfinite(data):
    with pytest.raises(ValueError):
        normalize_audio(data, 32000, "cpu", pad_short=True)


@pytest.mark.parametrize("rate", [16000, 32000, 44100])
@pytest.mark.model
def test_opt_in_padding_matches_manual_normalized_input(rate, tmp_path, model_session):
    """Compare every native output to an independently normalized, right-padded input."""
    samples = rate // 10
    time = np.arange(samples, dtype=np.float32) / rate
    audio = np.stack([0.1 * np.sin(2 * np.pi * 440 * time), 0.05 * np.cos(2 * np.pi * 220 * time)])
    path = tmp_path / "short.wav"
    sf.write(path, audio.T, rate, subtype="FLOAT")
    manual = torch.from_numpy(audio).mean(dim=0)
    if rate != 32000:
        manual = resample(manual, rate, 32000)
    normalized = manual.numel()
    manual = torch.nn.functional.pad(manual, (0, 9920 - normalized))
    expected = model_session.infer(manual, sample_rate=32000)
    provenance = model_session.loaded_checkpoint_info()
    for source, sample_rate in [(audio, rate), (torch.from_numpy(audio), rate), (path, None)]:
        with pytest.raises(ValueError, match="at least"):
            model_session.infer(source, sample_rate=sample_rate)
        actual = model_session.infer(source, sample_rate=sample_rate, pad_short=True)
        assert actual.scores == expected.scores
        assert actual.embedding == expected.embedding
        assert actual.audio.normalized_samples == normalized
        assert actual.audio.model_samples == 9920
        assert actual.audio.padding_samples == 9920 - normalized
        assert actual.audio.padding_seconds == (9920 - normalized) / 32000
        assert actual.audio.samples == samples
        assert actual.audio.channels == 2
        assert actual.audio.duration_seconds == samples / rate
        assert model_session.status == "ready"
    assert model_session.loaded_checkpoint_info() == provenance
    normal_path = ROOT / "audio/silence_32000_1ch.wav"
    assert model_session.infer(normal_path, pad_short=True) == model_session.infer(normal_path)
