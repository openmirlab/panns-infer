"""One validated file/waveform normalization boundary for PANNs CNN14.

File metadata is authoritative; arrays use channel-first shape and explicit rate.
Float32 channel averaging precedes resampling; inputs are never silently padded.
Reads: SoundFile audio paths, NumPy/Torch waveforms and their sample rates.
"""

from pathlib import Path

import numpy as np
import torch
from torchaudio.functional import resample

from ..results import AudioInfo
from .model import MIN_SAMPLES, SAMPLE_RATE


def normalize_audio(audio, sample_rate, device):
    if isinstance(audio, (str, Path)):
        import soundfile as sf

        if not Path(audio).is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio}")
        data, rate = sf.read(str(audio), dtype="float32", always_2d=True)
        if sample_rate is not None and sample_rate != rate:
            raise ValueError("sample_rate disagrees with file metadata")
        sample_rate = int(rate)
        waveform = torch.from_numpy(data.T.copy())
    elif isinstance(audio, np.ndarray):
        if not np.issubdtype(audio.dtype, np.floating):
            raise ValueError("Waveforms must contain floating point samples")
        waveform = torch.from_numpy(np.asarray(audio).copy())
    elif isinstance(audio, torch.Tensor):
        waveform = audio
    else:
        raise TypeError("Audio must be a file path, numpy.ndarray or torch.Tensor")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError("A positive integer sample_rate is required")
    if waveform.ndim not in (1, 2):
        raise ValueError("Expected (samples,) or channel-first (channels, samples)")
    if any(s == 0 for s in waveform.shape):
        raise ValueError("Audio cannot be empty")
    if not waveform.is_floating_point():
        raise ValueError("Waveforms must contain floating point samples")
    if not torch.isfinite(waveform).all():
        raise ValueError("Audio contains NaN or infinite samples")
    channels = 1 if waveform.ndim == 1 else waveform.shape[0]
    samples = waveform.shape[-1]
    info = AudioInfo(sample_rate, channels, samples, samples / sample_rate)
    # Normalize on CPU like the recorded upstream-model input; model/device
    # transfer follows resampling, keeping file/array/device input consistent.
    waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    if sample_rate != SAMPLE_RATE:
        waveform = resample(waveform, sample_rate, SAMPLE_RATE)
    if waveform.numel() < MIN_SAMPLES:
        raise ValueError(
            f"CNN14 requires at least {MIN_SAMPLES} normalized samples (~0.31 seconds)"
        )
    return waveform[None].to(device), info
