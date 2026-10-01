# panns-infer — maintainer context

Independent modern inference package for original PANNs CNN14 AudioSet tagging
and native 2,048-dimensional clip embeddings. Original model/research belongs
to Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang and Mark D.
Plumbley. Credit the original audioset_tagging_cnn repository and PANNs paper;
code MIT, official Zenodo3987831 weights CC BY4.0. No bundled weights.

Only 32kCNN14/527classes is supported. Inputs use sf float32 channel mean then
torchaudio32kresample. Historical librosa file resampling is not a parity claim.
Native embedding is post-fc1ReLU, not L2 normalized. Every component is exported
in API/CLI JSON. No training/evaluation/datasets/notebooks or upstream tree ship.

Status: unpublished private implementation. Original Torch1.13 CPU/CUDA golden
fixtures committed before extracting the core. Model/facade verified under
Torch2.7.1 CPU/CUDA with NumPy2.2.6. No container resource budget or new quality
study is claimed. CPU-only CI covers Python3.10–3.13; MPS is not supported.

Architecture: src/panns_infer lazy facade; session owns model/device/lifecycle;
checkpoints owns validated package config/downloader/cache/integrity; _impl
owns audio normalization/model graph; results owns portable output; cli calls
that same facade. No global resident cache or shared production runtime.

Load-bearing files start with title/rationale/Reads module docstrings. Keep
modules cohesive, graph/key layout faithful and configuration in one owner.
README and this file move with public behavior; CHANGELOG records changes.

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
Use work branches; main moves by merge. No publishing or weight rehosting.

Public skills check: parent inspected openmirlab-skills; its capability map
explicitly excludes private/unreleased packages. No public package entry is
appropriate until release; revisit install/task guidance in that same release
window. No constitution amendment is silently bundled with this package.
