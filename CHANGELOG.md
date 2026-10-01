# Changelog

## Unreleased — 0.1.0

- Extract original CNN14 AudioSet tagging without training/runtime framework dependencies.
- Preserve all527native scores and full2048native embedding with explicit meaning/provenance.
- Add independent lazy-import sessions, explicit device selection and tested release/reload.
- Add verified official CC-BY4checkpoint cache, offline/local/customregistry support.
- Add fullJSONCLI `panns-infer` and shared validated file/NumPy/Tensor inputs.
- Record original Torch1.13 CPU/CUDA golden fixtures before porting; verify modern2.7/NumPy2.
- Define explicit sf/torchaudio audio normalization; historical librosa resampling not claimed.
- No PyPI release, model rehosting, training/evaluation framework or dataset distribution.
