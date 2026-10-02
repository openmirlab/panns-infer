# Original-model baseline

The read-only reference is `qiuqiangkong/audioset_tagging_cnn`, commit
`d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4`. The reference's unchanged `Cnn14`
class is instantiated with 32,000 Hz, window 1,024, hop 320, 64 mel bins,
50–14,000 Hz and 527 classes, then its official checkpoint is loaded strictly.

Five generated CC0 signals cover silence, low noise, tonal sequences, modulated
harmonics and changing noise events; rates 16,000–48,000 Hz and mono/stereo.
This is numerical regression coverage, not a real-world tagging quality study.

Before extracting the production core, record waveform, logmel, all 527
clipwise scores and the native 2,048-component embedding twice. Each duplicate
run must be bit-identical. Baseline CPU and CUDA results have their own
`environment.json` and must not be cross-compared as bit-exact digests.

```bash
python tools/capture_baseline.py --upstream /path/to/read-only/upstream \
  --checkpoint /path/to/Cnn14.pth --device cpu
python tools/capture_baseline.py --upstream /path/to/read-only/upstream \
  --checkpoint /path/to/Cnn14.pth --device cuda:0
```

The file-input normalization contract is float32 decoding with SoundFile,
channel averaging and torchaudio functional resampling to 32 kHz. The baseline
records those normalized waveforms and sends them into the unchanged upstream
model. Historical `librosa.load` resampling is deliberately not a claim of
this package; upstream librosa 0.6.3 and modern librosa default to different
resamplers. Model parity and file-normalization policy are separate claims.

Fixtures never include private recordings, machine paths, weights or a model
state dictionary. The fixture manifest records license and content hashes.

## TorchAudio dependency extraction

The runtime now uses the unchanged three-function pure-Torch sinc resampling
closure from `pytorch/audio` revision
`95c61b4168fc5133be8dd8c1337d929d066ae6cf` (TorchAudio 2.7.1).
The BSD-2-Clause source and full license are retained; no TorchAudio binaries
are required for installing, testing or running the package.

Before extraction, `tools/capture_resample_baseline.py` recorded 20 seeded CC0
float32 inputs per CPU/CUDA device: five source rates (8, 16, 22.05, 44.1 and
48 kHz), mono/stereo and short/minimum-boundary lengths. Both original runs
were bit-exact. Input/output arrays, recording environment, pinned source and
fixture hashes live in `tests/fixtures/baseline_resample_{cpu,cuda}`.
`tests/test_resample.py` validates those originals on matching environments
and optionally compares unchanged TorchAudio 2.7.1 source plus every output
bit-exactly on current CPU/CUDA, using both float32 and float64.

Regeneration is reference-only: use a separate Torch 2.7.1 environment with
matching `torchaudio==2.7.1` installed explicitly. Neither baseline tool
belongs to the wheel or the runtime dependencies.

```bash
python tools/capture_resample_baseline.py --device cpu
python tools/capture_resample_baseline.py --device cuda:0
```
