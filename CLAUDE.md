# panns-infer — maintainer context

Independent modern inference package for original PANNs CNN14 AudioSet tagging
and native 2,048-dimensional clip embeddings. Original model/research belongs
to Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang and Mark D.
Plumbley. Credit the original audioset_tagging_cnn repository and PANNs paper;
PANNs/package code MIT, extracted TorchAudio sinc functions BSD-2-Clause;
combined code MIT AND BSD-2-Clause, official Zenodo3987831 weights CC BY4.0.
No bundled weights.

Only 32kCNN14/527classes is supported. Inputs use sf float32 channel mean then
unchanged TorchAudio2.7.1 pure-Torch sinc functions to32k, without TorchAudio
runtime dependency. Historical librosa file resampling is not a parity claim.
Native embedding is post-fc1ReLU, not L2 normalized. Every component is exported
in API/CLI JSON. No training/evaluation/datasets/notebooks or upstream tree ship.

Status: public GitHub repository since 2026-11-12, not on PyPI. Install with
`pip install 'panns-infer @ git+https://github.com/openmirlab/panns-infer.git'`.
The official Zenodo checkpoint remains CC BY 4.0 and is not bundled.
Original Torch1.13 CPU/CUDA golden
fixtures committed before extracting the core. Model/facade verified under
Torch2.7.1 CPU/CUDA with NumPy2.2.6. No container resource budget or new quality
study is claimed. CPU-only CI covers Python3.10–3.13; MPS is not supported.
Python3.13/Torch2.13+cpu without TorchAudio also passes full eligible model
tests (72passed,12recording/reference-only skips at that checkpoint).
Fresh Python3.13.5/Torch2.13.0+cpu paired installed wheels for this package and
beats-audioset-infer resolve core dependencies without TorchAudio. Installed
PANNs E2E passes all five original fixtures plus short manual-padding/CLI,
loaded provenance and release/reload checks. This is CPU evidence only.

Final local CUDA13.0/Ubuntu24.04 container verification: Python3.13.15,
Torch2.13.0+cu130, NumPy2.5.3, RTX4090/driver595.84. With TorchAudio imports
actively blocked, full cached GPU tests74passed/12explicit optional-reference
or recording-build skips; installed-wheel E2E passed five original fixtures,
full API/CLI scores/embeddings, short padding, provenance and release/reload.
Synthetic1/10/30/60/120second execution feasibility also passed, with weakref
model release and allocated memory returning to the independent framework
baseline. These are not quality/latency/wholeGPU/hostRAM or deployment budgets;
see docs/architecture.md. The container process terminated after verification.

Architecture: src/panns_infer lazy facade; session owns model/device/lifecycle;
checkpoints owns validated package config/downloader/cache/integrity; _impl
owns audio normalization/model graph; results owns portable output; cli calls
that same facade. No global resident cache or shared production runtime.

Load-bearing files start with title/rationale/Reads module docstrings. Keep
modules cohesive, graph/key layout faithful and configuration in one owner.
README and this file move with public behavior; CHANGELOG records changes.

`loaded_checkpoint_info()` is a ready-only, locked, defensive copy of the
successful verified load snapshot. It carries portable runtime identity,
repository/revisions, source/license metadata and artifact identities without
local paths; inspection does not consult files or the mutable checkpoint store.
Release/close clear the snapshot. `cache_info()` remains expected cache metadata.

Short input padding is opt-in (`infer(..., pad_short=True)` / `--pad-short`).
The single audio boundary right-pads only after existing mono/resampling to
9,920 samples; the default still rejects short input and empty/nonfinite input
always fails. `AudioInfo` preserves source duration and reports normalized/model
sample counts and added padding; padding is synthesized context, not coverage.
Manual normalized-padding A/B checks compare every score/embedding exactly.

`_impl/resample.py` contains the unchanged three-function TorchAudio2.7.1
closure (source95c61b4168fc5133be8dd8c1337d929d066ae6cf); its full BSD2 notice
ships in licenses. Exclude this verbatim file from formatting, not numerical
checks. Seeded CC0 original CPU/CUDA resample fixtures predate extraction;
`tests/test_resample.py` guards exact fixture comparisons by recording build and
hardware and optionally compares live source/results with TorchAudio2.7.1.
TorchAudio is needed only to regenerate original baselines, not package/tests.
Optional live-reference checks catch ImportError and OSError from absent or
incompatible TorchAudio binaries; these skips never disable package audio/model
checks. Tests simulate both failure types while exercising the package boundary.

Numerical changes require original golden evidence before edits. Bitwise
assertions only on matching recording environment; cross-version floats use
explicit tolerances, rankings/discrete outputs are checked separately. Real
model tests opt into local checkpoint paths and never implicitly download.
No model construction across repeated infer calls; release weak-reference and
reload tests are required. CPU thread/precision global settings belong to caller.

Exact verification commands from an activated development environment:

```bash
python -m pytest -q
PANNS_TEST_CHECKPOINT_DIR=/path/to/verified/assets PANNS_TEST_DEVICE=cpu python -m pytest -q
PANNS_TEST_CHECKPOINT_DIR=/path/to/verified/assets PANNS_TEST_DEVICE=cuda:0 python -m pytest -q
PYTHONPATH=src python tools/verify_parity.py --checkpoint /path/to/Cnn14.pth --device cpu
PYTHONPATH=src python tools/verify_parity.py --checkpoint /path/to/Cnn14.pth --device cuda:0
python -m ruff check .
python -m ruff format --check .
python -m build
python -m pip install dist/panns_infer-0.1.0-py3-none-any.whl
python -I tools/verify_installed_e2e.py --checkpoint-dir /path/to/verified/assets --device cpu
python -I tools/verify_installed_e2e.py --checkpoint-dir /path/to/verified/assets --device cuda:0
rg -n 'lightning|tensorboard|wandb|fairseq|torchlibrosa|librosa|dataset|optimizer' src pyproject.toml
```

Baseline regeneration imports the readonly original source via
`tools/capture_baseline.py`; never edit upstream to make tests pass. Research
baselines and synthetic assets belong in sdist/tests, never the wheel. Inspect
both archives and git-tracked files for weights/privatepaths/secrets before push.
Use work branches; main moves by merge. No PyPI publishing or weight rehosting.

Public skills check: the GitHub source is now public, so openmirlab-skills
task/install guidance should be updated in the same documentation window.
No constitution amendment is silently bundled with this package.
