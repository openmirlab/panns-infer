# Changelog

## Unreleased — 0.1.0

- Verify Python 3.13/Torch 2.13 CPU and CUDA 13.0 installed-wheel inference,
  embeddings, short padding and lifecycle without TorchAudio; synthetic inputs
  through 120 seconds pass execution-feasibility checks, without deployment or quality claims.

- Skip optional live-reference developer checks explicitly when TorchAudio is
  absent or its binary raises ImportError/OSError; mandatory package tests remain active.

- Remove the TorchAudio binary runtime dependency by preserving its unchanged
  BSD-2-Clause pure-Torch sinc resampling closure; record original CPU/CUDA
  fixtures and live bit-exact A/B evidence. Combined code license is now
  `MIT AND BSD-2-Clause`; full upstream attribution and license are retained.

- Add opt-in `pad_short=True` / `--pad-short` right-zero padding after existing
  mono/resampling, with normalized/model sample counts and explicit padding metadata;
  the default minimum-length rejection and normal-input model behavior are unchanged.

- Add ready-only `loaded_checkpoint_info()` with verified load identity and portable,
  defensively copied provenance; inspection never accesses files or loads a model.

- Extract original CNN14 AudioSet tagging without training/runtime framework dependencies.
- Preserve all527native scores and full2048native embedding with explicit meaning/provenance.
- Add independent lazy-import sessions, explicit device selection and tested release/reload.
- Add verified official CC-BY4checkpoint cache, offline/local/customregistry support.
- Add fullJSONCLI `panns-infer` and shared validated file/NumPy/Tensor inputs.
- Record original Torch1.13 CPU/CUDA golden fixtures before porting; verify modern2.7/NumPy2.
- Define explicit sf/torchaudio audio normalization; historical librosa resampling not claimed.
- No PyPI release, model rehosting, training/evaluation framework or dataset distribution.
